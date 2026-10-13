"""Choose a short temporary directory that supports multiprocessing sockets."""
import os
from multiprocessing import current_process
from pathlib import Path
import socket
import tempfile


def _probe_unix_sockets(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Match multiprocessing's actual pymp-*/listener-* pathname length.
    with tempfile.TemporaryDirectory(prefix='pymp-', dir=str(directory)) as temporary:
        address = str(Path(temporary) / 'listener-12345678')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(address)
            listener.listen(1)


def configure_ipc_tmpdir() -> Path:
    explicit = os.environ.get('FASTWAM_IPC_TMPDIR')
    if explicit:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = []
        inherited = os.environ.get('TMPDIR')
        if inherited:
            candidates.append(Path(inherited).expanduser().resolve())
        candidates += [Path(base) / f'fastwam-ipc-{os.getuid()}' for base in ('/tmp', '/var/tmp')]
    failures = []
    for directory in dict.fromkeys(candidates):
        try:
            _probe_unix_sockets(directory)
        except OSError as error:
            failures.append(f'{directory}: {error}')
            continue
        os.environ['TMPDIR'] = str(directory)
        os.environ['FASTWAM_IPC_TMPDIR'] = str(directory)
        tempfile.tempdir = None
        # multiprocessing caches its own pymp-* directory independently of
        # tempfile. Keep existing listeners alive, but let new listeners use
        # the selected directory after an environment/cache change.
        configuration = current_process()._config
        cached = configuration.get('tempdir')
        if cached:
            cached_path = Path(cached).resolve()
            if not cached_path.is_dir() or directory not in cached_path.parents:
                configuration.pop('tempdir', None)
        return directory
    raise RuntimeError(
        'No temporary directory can create multiprocessing Unix sockets. '
        'Set FASTWAM_IPC_TMPDIR to a short, writable directory on local disk '
        '(for example /tmp/fastwam-ipc-' + str(os.getuid()) + '). '
        + '; '.join(failures)
    )
