"""Download the exact resource snapshot into its original project-relative paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
BLOCK_SIZE = 8 * 1024 * 1024


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        redirected = super().redirect_request(request, fp, code, message, headers, new_url)
        if redirected is not None and urlsplit(request.full_url).netloc != urlsplit(new_url).netloc:
            redirected.remove_header('Authorization')
        return redirected


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b''):
            result.update(block)
    return result.hexdigest()


def safe_path(root, relative):
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or '..' in parsed.parts or not parsed.parts:
        raise ValueError('Invalid resource path: ' + relative)
    path = root.joinpath(*parsed.parts)
    # Files may be linked to a shared local snapshot, but directory aliases must
    # never redirect new downloads outside this project.
    parent = path.parent.resolve()
    if parent != root and root not in parent.parents:
        raise ValueError('Resource directory resolves outside the project: ' + relative)
    return path


def resource_present(root, row, verify=False):
    path = safe_path(root, row['path'])
    try:
        if not path.is_file() or path.stat().st_size != row['size']:
            return False
        return not verify or sha256(path) == row['sha256']
    except FileNotFoundError:
        return False


def fetch_local(root, row, source, link_local):
    original = source.joinpath(*PurePosixPath(row['path']).parts)
    target = safe_path(root, row['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not original.is_file() or original.stat().st_size != row['size']:
        raise RuntimeError('Local source is missing or has the wrong size: ' + row['path'])
    if link_local:
        # The source is the already validated complete snapshot. Users can use
        # --verify to additionally read and hash every file on shared storage.
        temporary = target.with_name(target.name + '.part-link')
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(original.resolve())
        temporary.replace(target)
        return
    temporary = target.with_name(target.name + '.part')
    digest = hashlib.sha256()
    with original.open('rb') as incoming, temporary.open('wb') as outgoing:
        for block in iter(lambda: incoming.read(BLOCK_SIZE), b''):
            digest.update(block)
            outgoing.write(block)
    if digest.hexdigest() != row['sha256']:
        temporary.unlink()
        raise RuntimeError('Local source checksum mismatch: ' + row['path'])
    shutil.copymode(original, temporary)
    temporary.replace(target)


def fetch_http(root, row, base_url, token):
    target = safe_path(root, row['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    for attempt in range(3):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > row['size']:
            temporary.unlink()
            offset = 0
        if temporary.exists() and offset == row['size']:
            if sha256(temporary) == row['sha256']:
                temporary.replace(target)
                return
            temporary.unlink()
            offset = 0
        headers = {'User-Agent': 'fastwam-rmbench-portable/1', 'Accept-Encoding': 'identity'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if offset:
            headers['Range'] = 'bytes=' + str(offset) + '-'
        url = base_url.rstrip('/') + '/' + quote(row['path'], safe='/')
        digest = hashlib.sha256()
        try:
            with build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=120) as incoming:
                resumed = offset and incoming.status == 206
                if resumed:
                    expected_range = 'bytes ' + str(offset) + '-'
                    if not incoming.headers.get('Content-Range', '').startswith(expected_range):
                        raise RuntimeError('Server returned an invalid resume offset')
                    with temporary.open('rb') as prefix:
                        for block in iter(lambda: prefix.read(BLOCK_SIZE), b''):
                            digest.update(block)
                else:
                    offset = 0
                written = offset
                with temporary.open('ab' if resumed else 'wb') as outgoing:
                    for block in iter(lambda: incoming.read(BLOCK_SIZE), b''):
                        written += len(block)
                        if written > row['size']:
                            raise RuntimeError('Server returned more bytes than the manifest size')
                        digest.update(block)
                        outgoing.write(block)
            if written != row['size']:
                raise RuntimeError('Incomplete download')
            if digest.hexdigest() != row['sha256']:
                temporary.unlink()
                raise RuntimeError('Downloaded checksum does not match the manifest')
            temporary.replace(target)
            return
        except HTTPError as error:
            if error.code in (401, 403):
                raise RuntimeError('Resource access denied. Set HF_TOKEN to a read token for this Hugging Face repository.') from None
            if error.code == 404:
                raise RuntimeError('Resource is not available yet on Hugging Face: ' + row['path']) from None
            if attempt == 2:
                raise RuntimeError('HTTP download failed for ' + row['path'] + ' (status ' + str(error.code) + ')') from None
        except Exception as error:
            if attempt == 2:
                raise RuntimeError('Download failed for ' + row['path'] + ': ' + type(error).__name__) from None
        time.sleep(1 + attempt)


def restore_links(root, links):
    for row in links:
        path = safe_path(root, row['path'])
        linked = Path(os.path.abspath(path.parent / row['target']))
        if linked != root and root not in linked.parents:
            raise ValueError('Snapshot alias would point outside the project: ' + row['path'])
        if path.exists() or path.is_symlink():
            if path.is_symlink() and os.readlink(path) == row['target']:
                continue
            raise RuntimeError('Refusing to replace an existing snapshot alias: ' + row['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(row['target'])


def restore_placeholders(root, rows):
    for row in rows:
        content = row['content'].encode('ascii')
        if len(content) != row['size'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Corrupt upstream reference metadata: ' + row['path'])
        if resource_present(root, row, verify=True):
            continue
        path = safe_path(root, row['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.part')
        temporary.write_bytes(content)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Override with a local snapshot directory or an HTTP base URL')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--check', action='store_true', help='Check file presence and size without downloading')
    parser.add_argument('--verify', action='store_true', help='Also hash already present files; reads the entire snapshot')
    parser.add_argument('--link-local', action='store_true', help='Use shared-storage file symlinks instead of copying local resources')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'assets/manifest.json')
    parser.add_argument('--destination', type=Path, default=ROOT)
    cli = parser.parse_args()
    if cli.workers < 1 or cli.workers > 8:
        parser.error('workers must be between 1 and 8')
    root = cli.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(cli.manifest.read_text())
    marker_dir = root / '.assets'
    marker_dir.mkdir(exist_ok=True)
    with (marker_dir / 'fetch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        missing = [row for row in manifest['files'] if not resource_present(root, row, cli.verify)]
        total = sum(row['size'] for row in missing)
        print('[assets] ' + str(len(missing)) + ' files required, ' + format(total / 1e9, '.2f') + ' GB', flush=True)
        if cli.check:
            missing += [row for row in manifest.get('placeholders', []) if not resource_present(root, row, cli.verify)]
            for row in missing[:12]:
                print('[missing] ' + row['path'])
            return 1 if missing else 0
        source = cli.source or os.environ.get('FASTWAM_ASSET_SOURCE')
        hub_path = ROOT / 'assets/hub.json'
        hub = json.loads(hub_path.read_text()) if hub_path.exists() else {}
        using_default_hub = False
        if missing and not source:
            if hub.get('repo_id'):
                revision = os.environ.get('FASTWAM_ASSET_REVISION', hub.get('revision', 'main'))
                source = 'https://huggingface.co/datasets/' + hub['repo_id'] + '/resolve/' + revision
                using_default_hub = True
            else:
                source = manifest.get('default_local_source')
        if missing and not source:
            parser.error('No Hugging Face resource repository or local source is configured')
        if missing:
            token = os.environ.get('HF_TOKEN')
            if not token:
                token_path = Path(os.environ.get('HF_TOKEN_PATH', str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'token')))
                if token_path.is_file():
                    token = token_path.read_text().strip()
            if using_default_hub:
                headers = {'User-Agent': 'fastwam-rmbench-portable/1'}
                if token:
                    headers['Authorization'] = 'Bearer ' + token
                try:
                    with build_opener(SafeRedirect()).open(Request(source + '/ASSETS_READY.json', headers=headers), timeout=30) as response:
                        uploaded = json.load(response)
                except HTTPError as error:
                    if error.code == 404:
                        raise RuntimeError('The Hugging Face resource upload has not completed yet; ASSETS_READY.json is missing.') from None
                    raise RuntimeError('Cannot read Hugging Face resource readiness metadata (HTTP ' + str(error.code) + ')') from None
                if uploaded.get('bundle_id') != manifest['bundle_id']:
                    raise RuntimeError('Hugging Face resource snapshot does not match this code repository')
            remote = source.startswith(('https://', 'http://'))
            if cli.link_local and remote:
                parser.error('--link-local requires a local shared-storage source')
            local = Path(source).expanduser().resolve() if not remote else None
            if not remote and not local.is_dir():
                parser.error('The local resource source directory does not exist')
            done = 0
            with ThreadPoolExecutor(max_workers=cli.workers) as pool:
                pending = {pool.submit(fetch_http, root, row, source, token) if remote
                           else pool.submit(fetch_local, root, row, local, cli.link_local): row
                           for row in missing}
                try:
                    for future in as_completed(pending):
                        row = pending[future]
                        future.result()
                        done += 1
                        if done <= 3 or done % 50 == 0 or done == len(missing):
                            print('[assets] ' + str(done) + '/' + str(len(missing)) + ' ' + row['path'], flush=True)
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
        restore_placeholders(root, manifest.get('placeholders', []))
        restore_links(root, manifest.get('links', []))
        remaining = [row['path'] for row in manifest['files'] if not resource_present(root, row)]
        if remaining:
            raise RuntimeError('Resource restoration is incomplete: ' + remaining[0])
        ready = {'bundle_id': manifest['bundle_id'], 'manifest_sha256': sha256(cli.manifest),
                 'files': len(manifest['files']), 'bytes': manifest['total_bytes']}
        temporary = marker_dir / 'READY.tmp'
        temporary.write_text(json.dumps(ready, indent=2) + '\n')
        temporary.replace(marker_dir / 'READY')
        print('[assets] Complete. Resources are restored to the expected project directories.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print('[assets] ' + str(error), file=sys.stderr)
        raise SystemExit(1)
