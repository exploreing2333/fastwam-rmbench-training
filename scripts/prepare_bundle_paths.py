"""Resolve dataset locations at the package's current location."""
import json
import os
import sysconfig
from pathlib import Path

root = Path(os.environ['FASTWAM_BUNDLE_ROOT']).resolve()
site = Path(sysconfig.get_paths()['purelib'])
pth_path = site / 'fastwam_bundle.pth'
pth_content = '\n'.join(str(root / name) for name in
                         ('src', 'vendor/curobo_src', 'vendor/LIBERO')) + '\n'
if not pth_path.is_file() or pth_path.read_text() != pth_content:
    pth_path.write_text(pth_content)
source = json.loads((root / 'data/source_map_template.json').read_text())
for key in ('dataset_root', 'processed_root', 'raw_root'):
    source[key] = str(root / source[key])
for row in source['entries']:
    for key in ('source_hdf5', 'processed_hdf5'):
        row[key] = str(root / row[key])
runtime = Path(os.environ.get('FASTWAM_RUNTIME_ROOT', str(root / '.runtime'))).resolve()
path = runtime / 'rmbench_source_map.json'
temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
temporary.write_text(json.dumps(source, indent=2) + '\n')
temporary.replace(path)
