"""Strict ZIP structure checks and seek adapters; no extraction or image loading."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import PurePosixPath
import re
import stat
import struct
import tarfile
import types
import zipfile
from pr42_stream_io import BLOCK, CHUNK, SliceReader, require

MAX_JSON = 24 * CHUNK
MAX_METADATA = 128 * CHUNK


def extras(raw):
    result, offset = {}, 0
    while offset < len(raw):
        require(offset+4 <= len(raw), "Truncated ZIP extra header")
        kind, length = struct.unpack_from("<HH", raw, offset)
        offset += 4
        require(offset+length <= len(raw) and kind not in result, "Invalid ZIP extra field")
        result[kind] = raw[offset:offset+length]
        offset += length
    return result


def local_layout(base, member, central_start):
    base.seek(member.header_offset)
    raw = base.read(30)
    require(len(raw) == 30, "Truncated local header")
    signature, version, flags, method, _, _, crc, compressed, size, nlen, xlen = struct.unpack("<4s5H3I2H", raw)
    # upload-artifact starts streaming v20 local headers before ZIP64 is known;
    # central v45 then carries actual 64-bit sizes/offsets with flag-8 descriptors.
    version_ok = version == member.extract_version and version in (20, 45)
    version_ok |= (version == 20 and member.extract_version == 45 and flags & 8 != 0
                   and 1 in extras(member.extra))
    require(signature == b"PK\x03\x04" and version_ok and flags == member.flag_bits
            and method == member.compress_type, "Local/central header mismatch")
    require(flags & ~(8 | 2048) == 0, "Unsupported ZIP flags")
    name, extra_raw = base.read(nlen), base.read(xlen)
    require(len(name) == nlen and len(extra_raw) == xlen, "Truncated header fields")
    require(name.decode("utf-8" if flags & 2048 else "cp437") == member.orig_filename,
            "Local/central name mismatch")
    extra = extras(extra_raw)
    sentinel_size, sentinel_compressed = size == 0xffffffff, compressed == 0xffffffff
    if sentinel_size or sentinel_compressed:
        value = extra.get(1, b"")
        need = 8 * (sentinel_size + sentinel_compressed)
        require(len(value) >= need, "Missing local ZIP64 size")
        position = 0
        if sentinel_size:
            size = struct.unpack_from("<Q", value, position)[0]
            position += 8
        if sentinel_compressed:
            compressed = struct.unpack_from("<Q", value, position)[0]
    if flags & 8:
        require(crc in (0, member.CRC) and size in (0, member.file_size)
                and compressed in (0, member.compress_size), "Local deferred metadata mismatch")
    else:
        require((crc, compressed, size) == (member.CRC, member.compress_size, member.file_size),
                "Local CRC/size mismatch")
    start = member.header_offset + 30 + nlen + xlen
    end = start + member.compress_size
    require(start <= end <= central_start, "Member intrudes into central directory")
    if flags & 8:
        base.seek(end)
        descriptor = base.read(min(24, central_start-end))
        signed = descriptor.startswith(b"PK\x07\x08")
        prefix = 4 if signed else 0
        matches = []
        for fmt in ("<III", "<IQQ"):
            length = struct.calcsize(fmt)
            if len(descriptor) >= prefix+length:
                fields = struct.unpack_from(fmt, descriptor, prefix)
                if fields == (member.CRC, member.compress_size, member.file_size):
                    matches.append(prefix+length)
        require(len(matches) == 1, "Invalid or ambiguous ZIP data descriptor")
        end += matches[0]
        require(end <= central_start, "Descriptor overlaps central directory")
    return start, end


def open_checked(base):
    zipped = zipfile.ZipFile(base)
    try:
        members = zipped.infolist()
        require(0 < len(members) <= 4096, "ZIP member limit")
        require(0 <= zipped.start_dir < base.size, "Central directory offset")
        names, layouts = set(), {}
        expanded = metadata = 0
        for member in members:
            name = member.orig_filename
            require(name == member.filename and "\0" not in name, "Normalized ZIP name")
            canonical = name[:-1] if member.is_dir() else name
            require(canonical and canonical not in names and "\\" not in name
                    and not PurePosixPath(canonical).is_absolute()
                    and all(x not in ("", ".", "..") for x in canonical.split("/")), "Unsafe/duplicate ZIP path")
            require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR),
                    "Special ZIP member")
            require(member.compress_type == zipfile.ZIP_STORED
                    and member.compress_size == member.file_size, "Only exact stored ZIP members supported")
            require(0 <= member.file_size <= 12*1024**3 and member.header_offset >= 0,
                    "ZIP member size/offset limit")
            if member.is_dir():
                require(member.file_size == 0 and member.CRC == 0, "Invalid empty ZIP directory")
            names.add(canonical)
            expanded += member.file_size
            if name != "candidate.tar":
                metadata += member.file_size
                require(member.file_size <= MAX_JSON, "Metadata member limit")
            layouts[name] = local_layout(base, member, zipped.start_dir)
        require(expanded < 12*1024**3 and metadata <= MAX_METADATA, "Expanded package limit")
        ordered = sorted(members, key=lambda x: x.header_offset)
        for prior, following in zip(ordered, ordered[1:]):
            require(layouts[prior.orig_filename][1] <= following.header_offset, "Overlapping ZIP members")
        require("candidate.tar" in layouts, "Missing candidate archive")
        return zipped, layouts
    except BaseException:
        zipped.close()
        raise


def member_bytes(zipped, name, limit=MAX_JSON):
    member = zipped.getinfo(name)
    require(not member.is_dir() and member.file_size <= limit, "Metadata size/type limit")
    with zipped.open(member) as stream:
        value = bytearray()
        while chunk := stream.read(CHUNK):
            value.extend(chunk)
            require(len(value) <= limit, "Metadata expanded limit")
    require(len(value) == member.file_size, "Metadata length mismatch")
    return bytes(value)


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def verify_inventory(zipped, package, budget):
    files = package.get("files")
    require(isinstance(files, dict) and files, "Missing package inventory")
    names = {x.filename for x in zipped.infolist() if not x.is_dir()}
    require(names == set(files) | {"approval-package.json"}, "Exact package file inventory mismatch")
    for name, expected in files.items():
        require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected), "Invalid member SHA")
        digest, count = hashlib.sha256(), 0
        with zipped.open(name) as stream:
            while data := stream.read(CHUNK):
                budget.check()
                digest.update(data)
                count += len(data)
        require(count == zipped.getinfo(name).file_size and digest.hexdigest() == expected,
                "Inventory member digest mismatch: " + name)


def candidate_identity(root, base, zipped, layouts):
    path = root / "scripts/candidate-archive-identity.py"
    spec = importlib.util.spec_from_file_location("pr42_canonical_candidate_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    size = zipped.getinfo("candidate.tar").file_size
    start = layouts["candidate.tar"][0]

    class MemberPath:
        def open(self, mode):
            require(mode == "rb", "Read-only member adapter")
            return SliceReader(base, start, size)
        def stat(self):
            return types.SimpleNamespace(st_size=size)

    @contextlib.contextmanager
    def open_tar(member, mode):
        require(mode == "r:", "Read-only canonical tar adapter")
        with member.open("rb") as stream:
            with tarfile.open(fileobj=stream, mode=mode) as archive:
                yield archive

    # Replace only this imported module's I/O namespace, not security logic or global tarfile.
    module.tarfile = types.SimpleNamespace(open=open_tar)
    return module.identity(MemberPath())
