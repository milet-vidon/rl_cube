"""Fail-closed path tests; no GPU or Isaac runtime required."""
import json
import tempfile
import unittest
from pathlib import Path
from evidence_paths import local_path,raw_sha


class EvidencePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.ws=Path(self.tmp.name);self.project=self.ws/'revotron_cube_rl';self.project.mkdir()

    def test_legacy_maps_to_current_clone_even_when_missing(self):
        p=local_path('/home/brainco/rl_cube/revotron_cube_rl/runs/example/result.json',self.project)
        self.assertEqual(p,self.project/'runs/example/result.json')
        with self.assertRaises(FileNotFoundError):raw_sha(p)

    def test_assets_map_to_clone(self):
        self.assertEqual(local_path('/home/brainco/rl_cube/revotron_revo3/model.usd',self.project),self.ws/'revotron_revo3/model.usd')

    def test_explicit_evidence_root(self):
        p=local_path('/home/brainco/rl_cube/revotron_cube_rl/runs/a/result.json',self.project,self.ws/'evidence')
        self.assertEqual(p,self.ws/'evidence/runs/a/result.json')

    def test_unknown_absolute_rejected(self):
        with self.assertRaises(ValueError):local_path('/home/someone/model.usd',self.project)

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):local_path('../old_project/result.json',self.project)

    def test_symlink_escape_rejected(self):
        (self.project/'escape').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):local_path('escape',self.project)

    def test_lfs_pointer_rejected(self):
        p=self.project/'model.pt';p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'0'*64+'\nsize 999\n')
        with self.assertRaisesRegex(RuntimeError,'LFS'):raw_sha(p)


if __name__=='__main__':unittest.main()
