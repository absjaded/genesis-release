"""Check exported file sizes and SHA-256 digests against the provenance manifest."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'PROVENANCE.json').read_text())
    for name, record in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('unsafe inventory path')
        raw = path.read_bytes()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('changed file: ' + name)
    print(f"Verified file sizes and SHA-256 digests for {len(manifest['files'])} exported files.")


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
