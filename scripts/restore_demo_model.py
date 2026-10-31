"""Restore the bundled M0 model from verified, exact-byte Git transport parts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'artifacts/m0_tfidf_logreg_retrained.parts/manifest.json'
DEFAULT_OUTPUT = ROOT / 'artifacts/m0_tfidf_logreg_retrained.joblib'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore_model(manifest_path: Path = DEFAULT_MANIFEST,
                  output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Validate an existing model or assemble parts without changing model bytes.

    An existing mismatched output is never overwritten. A partial or corrupt
    assembly is removed before returning an error.
    """
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    if not manifest_path.is_file():
        raise ValueError(f'Model manifest not found: {manifest_path}. Obtain the complete repository.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or not manifest.get('parts'):
        raise ValueError('Unsupported or empty model manifest.')
    expected_size, expected_sha = manifest['size'], manifest['sha256']
    if output_path.exists():
        if (output_path.is_file() and output_path.stat().st_size == expected_size
                and sha256_file(output_path) == expected_sha):
            return output_path
        raise ValueError(f'Existing model failed size/SHA-256 verification: {output_path}. '
                         'Move it aside before restoring; it has not been overwritten.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        total_hash, total_size = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(mode='wb', prefix=output_path.name + '.',
                                         suffix='.tmp', dir=output_path.parent,
                                         delete=False) as target:
            temporary = Path(target.name)
            for entry in manifest['parts']:
                name = entry['file']
                if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
                    raise ValueError(f'Invalid model part name: {name!r}')
                part = manifest_path.parent / name
                if not part.is_file():
                    raise ValueError(f'Model part missing: {part}. Obtain the complete repository.')
                part_hash, part_size = hashlib.sha256(), 0
                with part.open('rb') as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(chunk)
                        part_hash.update(chunk)
                        total_hash.update(chunk)
                        part_size += len(chunk)
                        total_size += len(chunk)
                if part_size != entry['size'] or part_hash.hexdigest() != entry['sha256']:
                    raise ValueError(f'Model part failed size/SHA-256 verification: {part}')
            if total_size != expected_size or total_hash.hexdigest() != expected_sha:
                raise ValueError('Restored model failed overall size/SHA-256 verification.')
            target.flush()
            os.fsync(target.fileno())
        # Publish atomically without overwriting even if another process wins.
        # The temporary file is on the same filesystem as the destination.
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise ValueError(f'Model appeared during restoration: {output_path}. '
                             'Retry to verify it; it has not been overwritten.') from error
        return output_path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        path = restore_model(args.manifest, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Verified M0 model: {path}')


if __name__ == '__main__':
    main()
