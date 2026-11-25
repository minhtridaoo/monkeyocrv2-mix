"""Discovery of the common pipeline after overlaying the Janek tree."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from mixab import monkey_janek as monkey


class LayoutTest(unittest.TestCase):
    def test_combined_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'rev4').mkdir()
            (root/'lowres').mkdir()
            with patch.object(monkey, 'WORKSPACE', root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(monkey.monkey_root(), root.resolve())

    def test_explicit_reference_has_priority(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = root/'own', root/'reference'
            for p in (a,b):
                (p/'rev4').mkdir(parents=True)
                (p/'lowres').mkdir()
            with patch.object(monkey, 'WORKSPACE', a), patch.dict(os.environ, {'MONKEY_V2_ROOT': str(b)}):
                self.assertEqual(monkey.monkey_root(), b.resolve())
