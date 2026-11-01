"""Bounded, digest-anchored read-only GitHub artifact transport."""
import collections
import hashlib
import io
import json
import os
import re
import selectors
import subprocess
import time

CHUNK = 1024**2
BLOCK = 8 * CHUNK
MAX_WIRE = 24 * 1024**3
MAX_REQUESTS = 2000
MAX_RSS = 512 * CHUNK
RANGE_ATTEMPTS = 2
REPO = "DataScience-EngineeringExperts/privatestack-bischat"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + 1800
        self.wire = 0
        self.requests = 0

    def check(self):
        require(time.monotonic() < self.deadline, "Global verification deadline")
        # No host/cgroup/RLIMIT changes; resource violations are failed evidence.
        with open("/proc/self/statm", encoding="ascii") as stream:
            rss = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        require(rss <= MAX_RSS, "Verifier RSS budget exceeded")
        with open("/proc/meminfo", encoding="ascii") as stream:
            available = next(int(line.split()[1])*1024 for line in stream if line.startswith("MemAvailable:"))
        require(available >= 1024**3, "Host RAM reserve exhausted")

    def count(self, count):
        self.wire += count
        require(self.wire <= MAX_WIRE, "Wire budget exceeded")
        self.check()


class ProcessStream:
    """Never expose command stderr, HTTP headers or signed URLs in failures."""
    def __init__(self, command, budget, timeout=30):
        budget.requests += 1
        require(budget.requests <= MAX_REQUESTS, "Request budget exceeded")
        budget.check()
        env = dict(os.environ)
        env.pop("GH_DEBUG", None)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.budget = budget
        self.deadline = min(time.monotonic() + timeout, budget.deadline)
        self.buffer = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.selector.close()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdout.close()

    def raw(self, size):
        self.budget.check()
        remaining = self.deadline - time.monotonic()
        require(remaining > 0 and bool(self.selector.select(min(remaining, 30))),
                "Download read timeout")
        data = os.read(self.process.stdout.fileno(), min(size, CHUNK))
        self.budget.count(len(data))
        return data

    def read(self, size):
        require(0 <= size <= BLOCK, "Read size exceeds bound")
        result = bytearray()
        if self.buffer:
            take = min(size, len(self.buffer))
            result.extend(self.buffer[:take])
            del self.buffer[:take]
        while len(result) < size:
            data = self.raw(size - len(result))
            if not data:
                break
            result.extend(data)
        return bytes(result)

    def headers(self):
        # gh --include emits only final response headers. Any other shape fails.
        while b"\r\n\r\n" not in self.buffer and b"\n\n" not in self.buffer:
            require(len(self.buffer) < 65536, "HTTP header limit")
            data = self.raw(4096)
            require(bool(data), "Missing HTTP headers")
            self.buffer.extend(data)
        separator = b"\r\n\r\n" if b"\r\n\r\n" in self.buffer else b"\n\n"
        raw, remainder = bytes(self.buffer).split(separator, 1)
        require(len(raw) <= 65536, "HTTP header limit")
        self.buffer = bytearray(remainder)
        lines = raw.decode("ascii", errors="strict").splitlines()
        match = re.fullmatch(r"HTTP/[0-9.]+ ([0-9]{3})(?: .*)?", lines[0])
        require(match is not None, "Malformed HTTP status")
        headers = {}
        for line in lines[1:]:
            require(":" in line, "Malformed HTTP header")
            key, value = line.split(":", 1)
            key = key.strip().lower()
            require(key not in headers, "Duplicate HTTP header")
            headers[key] = value.strip()
        return int(match.group(1)), headers

    def finish(self):
        require(self.read(1) == b"", "Oversized response")
        require(self.process.wait(timeout=max(.1, min(5, self.deadline-time.monotonic()))) == 0,
                "GitHub transport failed")


def api(path, budget):
    with ProcessStream(["gh", "api", "repos/" + REPO + path], budget) as stream:
        data = stream.read(4 * CHUNK)
        stream.finish()
    return json.loads(data)


def anchor(artifact_id, size, expected_digest, budget):
    require(type(size) is int and 0 < size <= 12 * 1024**3, "Artifact size limit")
    digest, chunks, count = hashlib.sha256(), [], 0
    with ProcessStream(["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact_id}/zip"],
                       budget, timeout=1800) as stream:
        while count < size:
            data = stream.read(min(CHUNK, size-count))
            require(len(data) == min(CHUNK, size-count), "Truncated full artifact")
            digest.update(data)
            chunks.append(hashlib.sha256(data).digest())
            count += len(data)
        stream.finish()
    require("sha256:" + digest.hexdigest() == expected_digest, "Full ZIP digest mismatch")
    return chunks


class HttpRange:
    def __init__(self, artifact_id, size, budget):
        self.artifact_id, self.size, self.budget = artifact_id, size, budget

    def __call__(self, start, end):
        require(0 <= start <= end < self.size and end-start+1 <= BLOCK, "Range bound")
        command = ["gh", "api", f"repos/{REPO}/actions/artifacts/{self.artifact_id}/zip",
                   "--include", "--method", "GET", "-H", f"Range: bytes={start}-{end}",
                   "-H", "Accept-Encoding: identity"]
        for attempt in range(RANGE_ATTEMPTS):
            try:
                with ProcessStream(command, self.budget) as stream:
                    status, headers = stream.headers()
                    require(status == 206, "Server did not return partial content")
                    require(headers.get("content-range") == f"bytes {start}-{end}/{self.size}",
                            "Content-Range mismatch")
                    require(headers.get("content-encoding", "identity") == "identity", "Encoded range")
                    length = end-start+1
                    require("content-length" not in headers or headers["content-length"] == str(length),
                            "Content-Length mismatch")
                    data = stream.read(length)
                    require(len(data) == length, "Truncated range")
                    stream.finish()
                    return data
            except ValueError as error:
                # A byte count shortfall can be a transient artifact redirect loss.
                # Retry it once; every retried byte is still checked against anchor.
                if str(error) != "Truncated range" or attempt + 1 == RANGE_ATTEMPTS:
                    raise


class AnchoredReader(io.RawIOBase):
    def __init__(self, size, hashes, fetch, budget):
        require(len(hashes) == (size+CHUNK-1)//CHUNK, "Incomplete chunk map")
        self.size, self.hashes, self.fetch, self.budget = size, hashes, fetch, budget
        self.position, self.cache = 0, collections.OrderedDict()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def clear(self): self.cache.clear()

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "Invalid seek origin")
        value = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        require(0 <= value <= self.size, "Seek outside artifact")
        self.position = value
        return value

    def block(self, index):
        if index not in self.cache:
            first = (index//8)*8
            start, end = first*CHUNK, min(self.size, (first+8)*CHUNK)-1
            data = self.fetch(start, end)
            require(len(data) == end-start+1, "Range payload length mismatch")
            pieces = [data[x:x+CHUNK] for x in range(0, len(data), CHUNK)]
            for n, piece in enumerate(pieces, first):
                require(hashlib.sha256(piece).digest() == self.hashes[n],
                        "Range differs from authenticated ZIP chunk")
            self.cache.clear()
            self.cache.update((first+n, piece) for n, piece in enumerate(pieces))
        return self.cache[index]

    def read(self, size=-1):
        self.budget.check()
        if size is None or size < 0:
            size = self.size-self.position
        require(size <= BLOCK, "Unbounded archive read rejected")
        size = min(size, self.size-self.position)
        data = bytearray()
        while len(data) < size:
            index, offset = divmod(self.position, CHUNK)
            piece = self.block(index)
            take = min(size-len(data), len(piece)-offset)
            require(take > 0, "Nonprogressing archive read")
            data.extend(piece[offset:offset+take])
            self.position += take
        return bytes(data)


class SliceReader(io.RawIOBase):
    def __init__(self, base, start, size):
        require(0 <= start <= base.size and 0 <= size <= base.size-start, "Slice bounds")
        self.base, self.start, self.size, self.position = base, start, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "Invalid slice seek")
        value = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        require(0 <= value <= self.size, "Seek outside member")
        self.position = value
        return value

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size-self.position
        require(size <= BLOCK, "Unbounded member read rejected")
        size = min(size, self.size-self.position)
        self.base.seek(self.start+self.position)
        data = self.base.read(size)
        self.position += len(data)
        return data
