"""Single-download disk evidence handling; no reuse, cleanup, ranges or image loads."""
import contextlib
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tarfile
import types

from pr42_stream_io import BLOCK, CHUNK, ProcessStream, REPO, require
from pr42_stream_zip import open_checked, strict_json

RESERVE = 2 * 1024**3
MAX_EXPANDED = 12 * 1024**3


def headroom(path, required, disk_usage=shutil.disk_usage):
    require(type(required) is int and required > 0, "Invalid disk requirement")
    available = disk_usage(path).free
    require(available > required, "Insufficient disk headroom")
    return {"required_free_bytes_exclusive": required, "available_free_bytes": available}


def fresh_destination(path, size):
    require(path.parent == Path("/tmp") and path.name == "pr42-gate3-local-35292240387",
            "Unexpected scratch destination")
    require(not path.exists() and not path.is_symlink(), "Evidence directory already exists")
    report = headroom(path.parent, 3*size+RESERVE)
    path.mkdir(mode=0o700)
    return report


class CheckedFile:
    """Read-only seekable local file with bounded reads and cooperative budget."""
    def __init__(self, path, budget):
        require(path.is_file() and not path.is_symlink(), "Nonregular proof file")
        self.file = path.open("rb")
        self.size, self.budget = os.fstat(self.file.fileno()).st_size, budget

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.file.tell()
    def close(self): self.file.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    def seek(self, offset, whence=0):
        self.budget.check()
        require(whence in (0, 1, 2), "Invalid seek")
        value = offset + (self.tell() if whence == 1 else self.size if whence == 2 else 0)
        require(0 <= value <= self.size, "Seek outside local proof")
        return self.file.seek(value)

    def read(self, count=-1):
        self.budget.check()
        if count is None or count < 0:
            count = self.size-self.tell()
        require(count <= BLOCK, "Unbounded local proof read")
        return self.file.read(count)


def download(artifact, archive, budget):
    size = artifact["size_in_bytes"]
    require(type(size) is int and 0 < size < MAX_EXPANDED, "Invalid download size")
    digest, count = hashlib.sha256(), 0
    # Exclusive file creation; partial evidence remains on every failure.
    with archive.open("xb") as target:
        with ProcessStream(["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact['id']}/zip"],
                           budget, timeout=1800) as stream:
            while count < size:
                headroom(archive.parent, size-count+RESERVE)
                wanted = min(CHUNK, size-count)
                data = stream.read(wanted)
                require(len(data) == wanted, "Truncated artifact download")
                target.write(data)
                count += len(data)
                digest.update(data)
            stream.finish()
        target.flush()
        os.fsync(target.fileno())
    require(count == size and "sha256:"+digest.hexdigest() == artifact["digest"],
            "Downloaded ZIP digest/size mismatch")
    budget.check()


def extract_verified(archive, evidence, budget):
    require(not evidence.exists() and not evidence.is_symlink(), "Evidence extraction already exists")
    with CheckedFile(archive, budget) as raw:
        zipped, _ = open_checked(raw)
        with zipped:
            members = zipped.infolist()
            expanded = sum(x.file_size for x in members)
            require(expanded < MAX_EXPANDED, "Expanded ZIP limit")
            headroom(evidence.parent, expanded+RESERVE)
            evidence.mkdir(mode=0o700)
            remaining = expanded
            hashes = {}
            for member in members:
                target = evidence / member.filename
                require(evidence in target.parents, "Extraction path escapes evidence")
                for parent in reversed(target.parents):
                    if parent == evidence or evidence in parent.parents:
                        if not parent.exists():
                            parent.mkdir(mode=0o700)
                        require(parent.is_dir() and not parent.is_symlink(), "Unsafe extraction parent")
                if member.is_dir():
                    target.mkdir(mode=0o700, exist_ok=True)
                    require(target.is_dir() and not target.is_symlink(), "Unsafe directory member")
                    continue
                digest, count = hashlib.sha256(), 0
                with zipped.open(member) as source, target.open("xb") as output:
                    while data := source.read(CHUNK):
                        budget.check()
                        require(count+len(data) <= member.file_size, "Expanded member overflow")
                        headroom(evidence, remaining+RESERVE)
                        output.write(data)
                        digest.update(data)
                        count += len(data)
                        remaining -= len(data)
                require(count == member.file_size, "Extracted member size mismatch")
                hashes[member.filename] = digest.hexdigest()
            require(remaining == 0, "Expanded package size mismatch")
    package_file = evidence / "approval-package.json"
    require(package_file.stat().st_size <= CHUNK, "Oversized approval package")
    package = strict_json(package_file.read_bytes())
    files = package.get("files")
    require(isinstance(files, dict) and files and set(hashes) == set(files) | {"approval-package.json"},
            "Exact file inventory mismatch")
    require(all(hashes.get(name) == value for name, value in files.items()), "Extracted inventory hash mismatch")
    actual = set()
    for path in evidence.rglob("*"):
        require(not path.is_symlink(), "Symlink in extracted evidence")
        if path.is_file(): actual.add(path.relative_to(evidence).as_posix())
    require(actual == set(hashes), "Unexpected extracted file")
    budget.check()
    return package, hashes["approval-package.json"]


def read_json(evidence, name, budget, limit=24*CHUNK):
    path = evidence / name
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "Invalid metadata file")
    with CheckedFile(path, budget) as stream:
        raw = bytearray()
        while data := stream.read(CHUNK):
            raw.extend(data)
            require(len(raw) <= limit, "Metadata read overflow")
    result = strict_json(raw)
    budget.check()
    return result


def local_identity(root, candidate, budget):
    """Unchanged canonical identity function; only its file access gains budgets."""
    spec = importlib.util.spec_from_file_location("pr42_disk_canonical_identity", root / "scripts/candidate-archive-identity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class BudgetedPath:
        def open(self, mode):
            require(mode == "rb", "Read-only canonical identity")
            return CheckedFile(candidate, budget)
        def stat(self): return candidate.stat()
    @contextlib.contextmanager
    def open_tar(path, mode):
        require(mode == "r:", "Read-only canonical archive")
        with path.open("rb") as stream:
            with tarfile.open(fileobj=stream, mode=mode) as archive:
                yield archive
    module.tarfile = types.SimpleNamespace(open=open_tar)
    result = module.identity(BudgetedPath())
    budget.check()
    return result
