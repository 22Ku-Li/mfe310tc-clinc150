"""Check exact restoration and safe failure for damaged model transport files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.restore_demo_model import restore_model


class RestoreDemoModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = bytes(range(256)) * 3
        parts = [self.payload[:500], self.payload[500:]]
        entries = []
        for i, content in enumerate(parts):
            name = f'part-{i:03d}.bin'
            (self.root / name).write_bytes(content)
            entries.append({'file': name, 'size': len(content),
                            'sha256': hashlib.sha256(content).hexdigest()})
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({
            'format_version': 1, 'size': len(self.payload),
            'sha256': hashlib.sha256(self.payload).hexdigest(), 'parts': entries,
        }))
        self.output = self.root / 'model.joblib'

    def test_exact_roundtrip_and_existing_reuse(self):
        restore_model(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), self.payload)
        before = self.output.stat().st_mtime_ns
        restore_model(self.manifest, self.output)
        self.assertEqual(self.output.stat().st_mtime_ns, before)

    def test_corrupt_or_missing_part_leaves_no_output(self):
        damaged = self.root / 'part-001.bin'
        damaged.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'verification'):
            restore_model(self.manifest, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.tmp')))
        damaged.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            restore_model(self.manifest, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_mismatched_existing_model_is_not_overwritten(self):
        self.output.write_bytes(b'keep me')
        with self.assertRaisesRegex(ValueError, 'not been overwritten'):
            restore_model(self.manifest, self.output)
        self.assertEqual(self.output.read_bytes(), b'keep me')


if __name__ == '__main__':
    unittest.main()
