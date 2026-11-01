"""Offline tests for the new local-disk verifier; no GitHub/API/artifact requests."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import pr42_disk_artifact as disk
from pr42_stream_io import Budget, CHUNK

spec = importlib.util.spec_from_file_location("disk_verifier", Path(__file__).with_name("pr42-verify-gate3-local-35292240387.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class FakeStream:
    data = b""
    def __init__(self, command, *_args, **_kwargs):
        if any("Range" in arg for arg in command): raise AssertionError("Range forbidden")
        self.stream = io.BytesIO(self.data)
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self, size): return self.stream.read(size)
    def finish(self):
        if self.stream.read(1): raise ValueError("oversized")


class DiskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pr42-disk-unit-")
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def archive(self, files=None, wrong_hash=False):
        files = files or {"candidate.tar": b"candidate", "scan.json": b"{}"}
        values = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
        if wrong_hash: values["candidate.tar"] = "0"*64
        package = {"files": values}
        path = self.root / "fixture.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zipped:
            for name, data in files.items(): zipped.writestr(name, data)
            zipped.writestr("approval-package.json", json.dumps(package))
        return path

    def test_exact_headroom_strict_greater(self):
        required = 3*verifier.SIZE+disk.RESERVE
        self.assertEqual(required, 21014531741)
        for available in (required-1, required):
            with self.assertRaises(ValueError):disk.headroom(self.root,required,lambda _:types.SimpleNamespace(free=available))
        self.assertEqual(disk.headroom(self.root,required,lambda _:types.SimpleNamespace(free=required+1))['available_free_bytes'],required+1)

    def test_wrong_or_existing_destination_rejected_without_writes(self):
        with self.assertRaises(ValueError):disk.fresh_destination(self.root, verifier.SIZE)
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'mkdir',side_effect=AssertionError('no mkdir')):
            with self.assertRaises(ValueError):disk.fresh_destination(verifier.DESTINATION,verifier.SIZE)

    def test_download_full_hash_and_exclusive_preservation(self):
        data=b"tiny artifact";target=self.root/'artifact.zip'
        artifact={'id':1,'size_in_bytes':len(data),'digest':'sha256:'+hashlib.sha256(data).hexdigest()}
        with patch.object(FakeStream,'data',data),patch.object(disk,'ProcessStream',FakeStream):
            disk.download(artifact,target,Budget())
            with self.assertRaises(FileExistsError):disk.download(artifact,target,Budget())
        self.assertEqual(target.read_bytes(),data)

    def test_download_truncated_oversized_or_wrong_digest_retained(self):
        artifact={'id':1,'size_in_bytes':3,'digest':'sha256:'+hashlib.sha256(b'abc').hexdigest()}
        for index,data in enumerate((b'ab',b'abcd',b'xyz')):
            target=self.root/str(index)
            with patch.object(FakeStream,'data',data),patch.object(disk,'ProcessStream',FakeStream):
                with self.assertRaises(ValueError):disk.download(artifact,target,Budget())
            self.assertTrue(target.exists())

    def test_extract_inventory_and_preserve_zip(self):
        archive=self.archive();evidence=self.root/'evidence'
        result,digest=disk.extract_verified(archive,evidence,Budget())
        self.assertEqual((evidence/'candidate.tar').read_bytes(),b'candidate')
        self.assertEqual(digest,hashlib.sha256((evidence/'approval-package.json').read_bytes()).hexdigest())
        self.assertTrue(archive.is_file());self.assertEqual(len(result['files']),2)
        with self.assertRaises(ValueError):disk.extract_verified(archive,evidence,Budget())

    def test_wrong_inventory_hash_preserves_partial_evidence(self):
        archive=self.archive(wrong_hash=True);evidence=self.root/'evidence'
        with self.assertRaises(ValueError):disk.extract_verified(archive,evidence,Budget())
        self.assertTrue((evidence/'candidate.tar').exists());self.assertTrue(archive.exists())

    def test_unsafe_zip_rejected_before_extraction(self):
        archive=self.archive({'candidate.tar':b'x','../escape':b'x'});evidence=self.root/'evidence'
        with self.assertRaises(ValueError):disk.extract_verified(archive,evidence,Budget())
        self.assertFalse(evidence.exists())

    def test_expansion_headroom_rejected_before_extraction(self):
        archive=self.archive();evidence=self.root/'evidence'
        with patch.object(disk,'headroom',side_effect=ValueError('disk')):
            with self.assertRaises(ValueError):disk.extract_verified(archive,evidence,Budget())
        self.assertFalse(evidence.exists())

    def test_local_read_budget_and_size(self):
        path=self.root/'data';path.write_bytes(b'abc')
        budget=Budget()
        with disk.CheckedFile(path,budget) as stream:
            with self.assertRaises(ValueError):stream.read(9*CHUNK)
            budget.deadline=0
            with self.assertRaises(ValueError):stream.read(1)

    def test_metadata_limit_and_invalid_json(self):
        (self.root/'data.json').write_bytes(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):disk.read_json(self.root,'data.json',Budget())
        with self.assertRaises(ValueError):disk.read_json(self.root,'data.json',Budget(),limit=1)

    def test_canonical_identity_exact_and_budgeted(self):
        config=json.dumps({'os':'linux','architecture':'amd64','config':{'Labels':{'org.opencontainers.image.revision':'fixture'}}}).encode()
        manifest=json.dumps([{'Config':'config.json','Layers':[],'RepoTags':['fixture:test']}]).encode()
        path=self.root/'candidate.tar'
        with tarfile.open(path,'w:') as archive:
            for name,data in [('manifest.json',manifest),('config.json',config)]:
                member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
        result=disk.local_identity(verifier.policy.DEFAULT_ROOT,path,Budget())
        self.assertEqual(result['archive_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['config_digest'],'sha256:'+hashlib.sha256(config).hexdigest())
        self.assertEqual(result['source_sha'],'fixture')
        budget=Budget();budget.deadline=0
        with self.assertRaises(ValueError):disk.local_identity(verifier.policy.DEFAULT_ROOT,path,budget)

    def package_fixture(self):
        config=json.dumps({'os':'linux','architecture':'amd64','config':{'Labels':{'org.opencontainers.image.revision':verifier.policy.SOURCE}}}).encode()
        image='sha256:'+hashlib.sha256(config).hexdigest()
        manifest=json.dumps([{'Config':'config.json','Layers':[],'RepoTags':['fixture:test']}]).encode()
        tar_bytes=io.BytesIO()
        with tarfile.open(fileobj=tar_bytes,mode='w:') as archive:
            for name,data in [('manifest.json',manifest),('config.json',config)]:
                member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
        candidate=tar_bytes.getvalue()
        identity={'archive_sha256':hashlib.sha256(candidate).hexdigest(),'image_ids':[image],'archive_size':len(candidate),
                  'config_digest':image,'repository_tags':['fixture:test'],'source_sha':verifier.policy.SOURCE,'layer_count':0}
        scans={'SchemaVersion':2,'Metadata':{'ImageID':image},'Results':[{'Target':'fixture','Class':'lang-pkgs','Type':'python','Vulnerabilities':[]}]}
        graph={'fixture':'a'*64}
        details={'GitPython':'3.1.59','transformers':'5.10.1','setuptools':'80.10.2','RestrictedPython':'8.5','unstructured':'0.24.0'}
        rows=[{'test':'pins_and_nltk_absent','status':'PASS','detail':details},
              {'test':'restrictedpython_guard','status':'PASS','detail':'ordinary compilation/access accepted; protected positional parameters, Formatter and async-generator inspection attributes denied'}]
        files={name:b'{}' for name in verifier.policy.REQUIRED}
        files.update({'candidate.tar':candidate,'archive-identity.json':json.dumps(identity).encode(),
                      'source-build-graph.json':json.dumps(graph).encode(),
                      'source-preparation-caller.json':json.dumps({'Account':'934824507893','Arn':'arn:aws:sts::934824507893:assumed-role/privatestack-bischat-security-scan/fixture'}).encode(),
                      'scan-checks.json':json.dumps({'schema_version':1,'configuration':'pass','secrets':'pass','sbom':'pass'}).encode(),
                      'session-feature-tests.log':('RESULTS_JSON='+json.dumps(rows)).encode()})
        files.update({name:json.dumps(scans).encode() for name in ('base.json','candidate.json','base-severity.json','candidate-severity.json')})
        package={'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},'schema_version':1,
                 'source_sha':verifier.policy.SOURCE,'workflow_run':str(verifier.policy.RUN),'repository_id':1309368517,
                 'preparation_mode':'source-proof-only','publication_ready':False,'scanned_image_id':image,'gate_exit_status':1}
        archive=self.root/'package.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as zipped:
            for name,data in files.items():zipped.writestr(name,data)
            zipped.writestr('approval-package.json',json.dumps(package))
        evidence=self.root/'extracted'
        disk.extract_verified(archive,evidence,Budget())
        return archive,evidence,package,graph

    def test_complete_local_package_and_caller_graph_identity_negatives(self):
        archive,evidence,package,graph=self.package_fixture()
        sys.path.insert(0,str(verifier.policy.DEFAULT_ROOT/'scripts'))
        with patch('source_archive_gate.graph',return_value=graph):
            identity,diagnostics,features=verifier.package_checks(verifier.policy.DEFAULT_ROOT,evidence,archive,package,Budget())
            self.assertEqual(features['RestrictedPython'],'8.5');self.assertFalse(features['full_filesystem_inventory_verified'])
            self.assertEqual(diagnostics['candidate_findings']['raw_finding_count'],0)
            for name,bad in [('source-build-graph.json',{'wrong':'a'*64}),('source-preparation-caller.json',{'Account':'wrong'}),
                             ('archive-identity.json',{'source_sha':'wrong'}),('candidate.json',{'SchemaVersion':2,'Results':[{}],'Metadata':{'ImageID':'wrong'}})]:
                path=evidence/name;original=path.read_bytes();path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):verifier.package_checks(verifier.policy.DEFAULT_ROOT,evidence,archive,package,Budget())
                path.write_bytes(original)


class AuthorityTests(unittest.TestCase):
    def payloads(self):
        repo={'id':1309368517};tree={'sha':'a'*40}
        pull={'number':42,'state':'open','head':{'sha':verifier.REVIEW,'repo':repo},'base':{'ref':'main','repo':repo}}
        return [pull,{'sha':verifier.policy.SOURCE,'tree':tree},{'sha':verifier.REVIEW,'tree':tree}]

    def test_pinned_artifact_and_equal_trees(self):
        artifact={'size_in_bytes':verifier.SIZE,'digest':verifier.DIGEST}
        with patch.object(verifier.policy,'authority',return_value=artifact),patch.object(verifier,'api',side_effect=self.payloads()),patch.object(verifier.policy,'git',side_effect=[verifier.REVIEW.encode(),b'a'*40]):
            self.assertEqual(verifier.authority(Path('/fixture'),Budget()),artifact)

    def test_artifact_and_review_substitutions_fail(self):
        for artifact in ({'size_in_bytes':1,'digest':verifier.DIGEST},{'size_in_bytes':verifier.SIZE,'digest':'wrong'}):
            with patch.object(verifier.policy,'authority',return_value=artifact):
                with self.assertRaises(ValueError):verifier.authority(Path('/fixture'),Budget())
        payloads=self.payloads();payloads[2]['tree']={'sha':'b'*40}
        with patch.object(verifier.policy,'authority',return_value={'size_in_bytes':verifier.SIZE,'digest':verifier.DIGEST}),patch.object(verifier,'api',side_effect=payloads),patch.object(verifier.policy,'git',side_effect=[verifier.REVIEW.encode(),b'a'*40]):
            with self.assertRaises(ValueError):verifier.authority(Path('/fixture'),Budget())

    def test_package_scope_rejected_before_file_reads(self):
        package={'files':dict.fromkeys(verifier.policy.REQUIRED),'source_sha':'wrong'}
        with self.assertRaises(ValueError):verifier.package_checks(Path('/fixture'),Path('/fixture'),Path('/fixture'),package,Budget())

    def test_diagnostics_still_bounded_and_non_authorizing(self):
        report={'Results':[{'Target':'t','Class':'lang-pkgs','Type':'python','Vulnerabilities':[{'Severity':'HIGH','PkgName':'p','InstalledVersion':'1','VulnerabilityID':'CVE-test'}]}]}
        result=verifier.policy.finding_summary(report)
        self.assertEqual(result['raw_finding_count'],1);self.assertFalse(result['dispositions_validated'])
        with self.assertRaises(ValueError):verifier.policy.serialize_result({'huge':'x'*(2*CHUNK)})

    def test_no_network_or_write_default(self):
        with patch.object(sys,'argv',['disk-verifier']),patch.object(verifier,'verify',side_effect=AssertionError('must not execute')):
            verifier.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)
