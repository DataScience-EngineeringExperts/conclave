"""Offline adversarial checks; no API, artifact download, disk archive or installation."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import pr42_stream_io as transport
import pr42_stream_zip as package
from pr42_stream_io import AnchoredReader, Budget, CHUNK, HttpRange, ProcessStream, SliceReader

ROOT = Path("/home/ubuntu/dev/privatestack-bischat-worktrees/pr42-retire-stale-pointer-20260918")


def reader(raw, fetch=None):
    hashes = [hashlib.sha256(raw[i:i+CHUNK]).digest() for i in range(0, len(raw), CHUNK)]
    return AnchoredReader(len(raw), hashes, fetch or (lambda a, b: raw[a:b+1]), Budget())


def zip_bytes(entries, method=zipfile.ZIP_STORED):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=method) as zipped:
        for name, data in entries:
            zipped.writestr(name, data)
    return result.getvalue()


def local_fixture(payload=b"abc", *, zip64=False, descriptor=False, signed=True):
    name = b"candidate.tar"
    crc, size = zlib.crc32(payload), len(payload)
    extra = struct.pack("<HHQQ", 1, 16, size, size) if zip64 else b""
    flags = 8 if descriptor else 0
    local_size = 0xffffffff if zip64 else 0 if descriptor else size
    header = struct.pack("<4s5H3I2H", b"PK\x03\x04", 45 if zip64 else 20,
                         flags, 0, 0, 0, 0 if descriptor else crc, local_size, local_size, len(name), len(extra))
    end = b""
    if descriptor:
        end = (b"PK\x07\x08" if signed else b"") + struct.pack("<IQQ" if zip64 else "<III", crc, size, size)
    raw = header+name+extra+payload+end
    member = zipfile.ZipInfo(name.decode())
    member.extract_version = 45 if zip64 else 20
    member.flag_bits, member.compress_type, member.CRC = flags, 0, crc
    member.file_size = member.compress_size = size
    member.header_offset = 0
    return raw, member


class RangeTests(unittest.TestCase):
    def test_chunk_map_and_last_partial(self):
        raw = b"a"*CHUNK+b"end"
        r = reader(raw)
        self.assertEqual(r.read(CHUNK), raw[:CHUNK])
        self.assertEqual(r.read(10), b"end")
        self.assertEqual(r.seek(-3, 2), CHUNK)
        self.assertEqual(r.read(), b"end")

    def test_toctou_byteflip_rejected_even_same_server_identity(self):
        raw = b"authenticated bytes"
        r = reader(raw, lambda a, b: b"X"+raw[1:])
        with self.assertRaises(ValueError): r.read(1)
        self.assertFalse(r.cache)

    def test_truncated_range_rejected(self):
        with self.assertRaises(ValueError): reader(b"abc", lambda a, b: b"ab").read(1)

    def test_seek_and_read_bounds(self):
        r = reader(b"abc")
        for position in (-1, 4):
            with self.assertRaises(ValueError): r.seek(position)
        with self.assertRaises(ValueError): r.read(transport.BLOCK+1)
        with self.assertRaises(ValueError): SliceReader(r, 0, 4)

    def test_slice_random_seek_without_preceding_download(self):
        raw = b"a"*(9*CHUNK)+b"end"
        calls = []
        def fetch(a, b): calls.append((a,b)); return raw[a:b+1]
        s = SliceReader(reader(raw, fetch), 9*CHUNK, 3)
        self.assertEqual(s.read(), b"end")
        self.assertEqual(calls, [(8*CHUNK, len(raw)-1)])

    def test_http_validation_matrix(self):
        for status, content_range, data in [(200,"bytes 0-2/3",b"abc"),
                (206,"bytes 0-2/4",b"abc"),(206,"bytes 1-3/3",b"abc"),(206,"bytes 0-2/3",b"ab")]:
            class Fake:
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
                def headers(self): return status,{"content-range":content_range,"x-location":"signed-secret"}
                def read(self,n): return data
                def finish(self): pass
            with patch.object(transport,"ProcessStream",Fake):
                with self.assertRaises(ValueError) as error: HttpRange(1,3,Budget())(0,2)
                self.assertNotIn("signed-secret",str(error.exception))

    def test_http_truncated_range_retried_once(self):
        calls = []
        class Fake:
            def __init__(self,*a,**k): calls.append(self)
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def headers(self): return 206,{"content-range":"bytes 0-2/3"}
            def read(self,n): return b"ab" if len(calls) == 1 else b"abc"
            def finish(self): pass
        with patch.object(transport,"ProcessStream",Fake):
            self.assertEqual(HttpRange(1,3,Budget())(0,2),b"abc")
        self.assertEqual(len(calls),2)

    def test_process_killed_on_non206(self):
        original = subprocess.Popen
        processes = []
        def start(*args, **kwargs):
            proc = original([sys.executable,"-c","import sys,time;sys.stdout.buffer.write(b'HTTP/2.0 200 OK\\r\\nLocation: signed-secret\\r\\n\\r\\n');sys.stdout.flush();time.sleep(30)"],**kwargs)
            processes.append(proc); return proc
        with patch.object(transport.subprocess,"Popen",side_effect=start):
            with self.assertRaises(ValueError): HttpRange(1,3,Budget())(0,2)
        self.assertIsNotNone(processes[0].returncode)

    def test_process_killed_on_timeout(self):
        stream = ProcessStream([sys.executable,"-c","import time;time.sleep(30)"],Budget(),timeout=.03)
        with self.assertRaises(ValueError):
            with stream: stream.read(1)
        self.assertIsNotNone(stream.process.returncode)

    def test_header_limit(self):
        stream=ProcessStream([sys.executable,"-c","import sys;sys.stdout.buffer.write(b'x'*70000)"],Budget())
        with self.assertRaises(ValueError):
            with stream: stream.headers()

    def test_late_full_zip_digest_failure(self):
        class Fake:
            def __init__(self,*a,**k): pass
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def read(self,n): return b"abc"
            def finish(self): pass
        with patch.object(transport,"ProcessStream",Fake):
            with self.assertRaises(ValueError):transport.anchor(1,3,"sha256:"+"0"*64,Budget())


class ZipTests(unittest.TestCase):
    def test_zip64_and_descriptors_all_forms(self):
        for use64 in (False, True):
            for descriptor in (False, True):
                for signed in (False, True):
                    raw, member = local_fixture(zip64=use64,descriptor=descriptor,signed=signed)
                    start,end=package.local_layout(reader(raw),member,len(raw))
                    self.assertEqual(end,len(raw));self.assertEqual(raw[start:start+3],b"abc")

    def test_zip64_size_above_4gb_without_large_fixture(self):
        raw,member=local_fixture(zip64=True)
        member.file_size=member.compress_size=2**32+3
        changed=bytearray(raw)
        offset=30+len(member.filename)+4
        struct.pack_into("<QQ",changed,offset,member.file_size,member.file_size)
        start=30+len(member.filename)+20
        _,end=package.local_layout(reader(bytes(changed)),member,start+member.file_size)
        self.assertEqual(end,start+member.file_size)

    def test_streamed_zip64_version_transition(self):
        raw,member=local_fixture(descriptor=True)
        member.extract_version=45;member.extra=struct.pack('<HHQQQ',1,24,3,3,0)
        self.assertEqual(package.local_layout(reader(raw),member,len(raw))[1],len(raw))
        member.extra=b''
        with self.assertRaises(ValueError):package.local_layout(reader(raw),member,len(raw))
        member.extract_version=63
        with self.assertRaises(ValueError):package.local_layout(reader(raw),member,len(raw))

    def test_bad_descriptor_and_local_crc(self):
        for descriptor in (False, True):
            raw,member=local_fixture(descriptor=descriptor)
            raw=bytearray(raw);raw[-1 if descriptor else 14]^=1
            with self.assertRaises(ValueError):package.local_layout(reader(bytes(raw)),member,len(raw))

    def test_local_central_names_flags_method_mismatch(self):
        for offset in (6,8,30):
            raw,member=local_fixture();raw=bytearray(raw);raw[offset]^=1
            with self.assertRaises((ValueError,NotImplementedError)):package.local_layout(reader(bytes(raw)),member,len(raw))

    def test_safe_inventory_every_member_hash(self):
        content=b"candidate bytes"
        raw=zip_bytes([("candidate.tar",content),("approval-package.json",b"{}")])
        z,_=package.open_checked(reader(raw))
        with z:package.verify_inventory(z,{"files":{"candidate.tar":hashlib.sha256(content).hexdigest()}},Budget())

    def test_wrong_member_hash_rejected(self):
        raw=zip_bytes([("candidate.tar",b"abc"),("approval-package.json",b"{}")])
        z,_=package.open_checked(reader(raw))
        with z:
            with self.assertRaises(ValueError):package.verify_inventory(z,{"files":{"candidate.tar":"0"*64}},Budget())

    def test_missing_inventory_member_rejected(self):
        z,_=package.open_checked(reader(zip_bytes([("candidate.tar",b"abc"),("extra",b"x")])) )
        with z:
            with self.assertRaises(ValueError):package.verify_inventory(z,{"files":{"candidate.tar":"0"*64}},Budget())

    def test_nul_path_and_traversal_rejected(self):
        for name in ("../candidate.tar","/candidate.tar","a\\candidate.tar"):
            with self.assertRaises(ValueError):package.open_checked(reader(zip_bytes([(name,b"x"),("candidate.tar",b"y")])))
        raw=zip_bytes([("candidate.tar",b"x"),("badXname",b"y")]).replace(b"badXname",b"bad\0name")
        with self.assertRaises(ValueError):package.open_checked(reader(raw))

    def test_deflated_bomb_rejected(self):
        with self.assertRaises(ValueError):package.open_checked(reader(zip_bytes([("candidate.tar",b"x"*CHUNK)],zipfile.ZIP_DEFLATED)))

    def test_zero_directory_crc_enforced(self):
        raw=bytearray(zip_bytes([("empty/",b""),("candidate.tar",b"x")]))
        struct.pack_into("<I",raw,14,123)
        central=raw.index(b"PK\x01\x02");struct.pack_into("<I",raw,central+16,123)
        with self.assertRaises(ValueError):package.open_checked(reader(bytes(raw)))

    def test_payload_intrusion(self):
        raw,member=local_fixture()
        with self.assertRaises(ValueError):package.local_layout(reader(raw),member,len(raw)-1)

    def test_duplicate_json_and_nonfinite(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):package.strict_json(raw)

    def test_canonical_identity_adapter(self):
        config=json.dumps({"os":"linux","architecture":"amd64","config":{"Labels":{"org.opencontainers.image.revision":"s"}}}).encode()
        manifest=json.dumps([{"Config":"config.json","RepoTags":["local:test"],"Layers":[]}]).encode()
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode="w:") as tar:
            for name,data in (("manifest.json",manifest),("config.json",config)):
                entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
        data=out.getvalue();r=reader(zip_bytes([("candidate.tar",data)]));z,layouts=package.open_checked(r)
        with z: result=package.candidate_identity(ROOT,r,z,layouts)
        self.assertEqual(result['archive_sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(result['config_digest'],'sha256:'+hashlib.sha256(config).hexdigest())
        self.assertEqual(result['archive_size'],len(data));self.assertEqual(result['source_sha'],'s')


class OrchestrationTests(unittest.TestCase):
    def module(self):
        path=Path(__file__).with_name('pr42-verify-gate3-streaming.py')
        spec=importlib.util.spec_from_file_location('verify_main',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_wrong_run_fails_before_download(self):
        m=self.module()
        with patch.object(m,'api',side_effect=[{}, {'head_sha':'wrong'}]):
            with self.assertRaises(ValueError):m.authority(1,ROOT,Budget())

    def test_runtime_83_missing_and_failed_guard_rejected(self):
        m=self.module()
        for rows in ([],[{'test':'pins_and_nltk_absent','status':'PASS','detail':{'RestrictedPython':'8.3'}}],
                     [{'test':'pins_and_nltk_absent','status':'FAIL'}]):
            raw=('RESULTS_JSON='+json.dumps(rows)).encode()
            with patch.object(m,'member_bytes',return_value=raw):
                with self.assertRaises(ValueError):m.runtime_features(None,reader(b'x'))

    def test_resource_budgets_fail_closed(self):
        b=Budget();b.deadline=0
        with self.assertRaises(ValueError):b.check()
        b=Budget();b.wire=transport.MAX_WIRE
        with self.assertRaises(ValueError):b.count(1)
        b=Budget();b.requests=transport.MAX_REQUESTS
        with self.assertRaises(ValueError):ProcessStream(['never-run'],b)
        for statm,mem in [('1 999999999','MemAvailable: 999999999 kB\n'),('1 1','MemAvailable: 1 kB\n')]:
            with patch('builtins.open',side_effect=[io.StringIO(statm),io.StringIO(mem)]):
                with self.assertRaises(ValueError):Budget().check()

    def test_anchor_truncation_and_oversize(self):
        original=subprocess.Popen
        for data in ('ab','abcd'):
            def start(*a,**kw):return original([sys.executable,'-c','import sys;sys.stdout.buffer.write('+repr(data.encode())+')'],**kw)
            with patch.object(transport.subprocess,'Popen',side_effect=start):
                with self.assertRaises(ValueError):transport.anchor(1,3,'sha256:'+hashlib.sha256(b'abc').hexdigest(),Budget())

    def test_http_length_and_oversize(self):
        original=subprocess.Popen
        for length,body in [('4',b'abc'),('3',b'abcd')]:
            response=b'HTTP/2.0 206 OK\r\nContent-Range: bytes 0-2/3\r\nContent-Length: '+length.encode()+b'\r\n\r\n'+body
            def start(*a,**kw):return original([sys.executable,'-c','import sys;sys.stdout.buffer.write('+repr(response)+')'],**kw)
            with patch.object(transport.subprocess,'Popen',side_effect=start):
                with self.assertRaises(ValueError):HttpRange(1,3,Budget())(0,2)

    def test_duplicate_and_overlapping_members(self):
        raw=zip_bytes([('candidate.tar',b'abc'),('candidate.tar',b'abc')])
        with self.assertRaises(ValueError):package.open_checked(reader(raw))
        raw=bytearray(zip_bytes([('candidate.tar',b'abc'),('second',b'def')]))
        second=raw.index(b'PK\x01\x02',raw.index(b'PK\x01\x02')+1)
        struct.pack_into('<I',raw,second+42,1)
        with self.assertRaises(ValueError):package.open_checked(reader(bytes(raw)))

    def test_no_network_default(self):
        path=Path(__file__).with_name('pr42-verify-gate3-streaming.py')
        spec=importlib.util.spec_from_file_location('verify_main',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(sys,'argv',['verify','--artifact-id','1']),patch.object(module,'verify',side_effect=AssertionError('must not execute')):
            module.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)
