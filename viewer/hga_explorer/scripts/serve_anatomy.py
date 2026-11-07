#!/usr/bin/env python3
"""Serve the anatomy review API, optionally together with the explorer dist.

Run with the ieeg conda environment (nibabel and matplotlib). Dev Vite proxies
``/api`` to this process. ``scripts/serve.sh`` passes ``--static dist`` so the
HPC port serves the built viewer and the API together.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

from anatomy_review import (  # noqa: E402
    contacts_payload,
    read_labels,
    render_slice_png,
    write_label,
)

mimetypes.add_type('model/gltf-binary', '.glb')
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('image/svg+xml', '.svg')
mimetypes.add_type('application/wasm', '.wasm')


class AnatomyServer(ThreadingHTTPServer):
    def __init__(self, address, handler, static_dir: Path | None):
        super().__init__(address, handler)
        self.static_dir = static_dir.resolve() if static_dir else None


class AnatomyHandler(BaseHTTPRequestHandler):
    server_version = 'AnatomyReview/1.0'

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith('/api/'):
            self._json(404, {'error': 'Not found'}, head_only=True)
            return
        self._serve_static(parsed.path, head_only=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith('/api/'):
            self._api_get(parsed)
            return
        self._serve_static(parsed.path, head_only=False)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/api/anatomy/labels':
            self._json(404, {'error': 'Not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self._json(400, {'error': 'Invalid Content-Length'})
            return
        if length < 0 or length > 65536:
            self._json(413, {'error': 'Body too large'})
            return
        raw = self.rfile.read(length) if length else b'{}'
        try:
            body = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {'error': 'Expected a JSON object'})
            return
        if not isinstance(body, dict):
            self._json(400, {'error': 'Expected a JSON object'})
            return
        try:
            payload = write_label(body.get('subject', ''), body.get('contact', ''), body.get('label'))
        except Exception as exc:
            self._error(exc)
            return
        self._json(200, payload)

    def _api_get(self, parsed):
        query = {}
        for part in parsed.query.split('&'):
            if not part:
                continue
            key, _, value = part.partition('=')
            query.setdefault(unquote(key), unquote(value))
        try:
            if parsed.path == '/api/anatomy/contacts':
                self._json(200, contacts_payload(query.get('subject', '')))
                return
            if parsed.path == '/api/anatomy/labels':
                subject = query.get('subject') or None
                self._json(200, {'labels': read_labels(subject)})
                return
            if parsed.path == '/api/anatomy/slice.png':
                png = render_slice_png(query.get('subject', ''), query.get('contact', ''))
                self._bytes(200, png, 'image/png')
                return
        except Exception as exc:
            self._error(exc)
            return
        self._json(404, {'error': 'Not found'})

    def _error(self, exc: Exception):
        if isinstance(exc, FileNotFoundError):
            self._json(404, {'error': str(exc)})
            return
        if isinstance(exc, LookupError):
            self._json(404, {'error': str(exc)})
            return
        if isinstance(exc, ValueError):
            self._json(400, {'error': str(exc)})
            return
        traceback.print_exc()
        self._json(500, {'error': 'Anatomy request failed'})

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json(self, status: int, payload: dict, head_only: bool = False):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _bytes(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self._cors()
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static(self, raw_path: str, head_only: bool):
        static_dir = self.server.static_dir
        if static_dir is None:
            self.send_error(404, 'No static directory')
            return
        request_path = unquote(raw_path.split('?', 1)[0])
        if request_path in ('', '/'):
            request_path = '/index.html'
        candidate = (static_dir / request_path.lstrip('/')).resolve()
        try:
            candidate.relative_to(static_dir)
        except ValueError:
            self.send_error(403)
            return
        if candidate.is_dir():
            candidate = (candidate / 'index.html').resolve()
        if not candidate.is_file():
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(str(candidate))
        size = candidate.stat().st_size
        self.send_response(200)
        self.send_header('Content-Type', mime or 'application/octet-stream')
        self.send_header('Content-Length', str(size))
        self.end_headers()
        if head_only:
            return
        with candidate.open('rb') as handle:
            while True:
                chunk = handle.read(1024 * 256)
                if not chunk:
                    break
                self.wfile.write(chunk)


def main():
    parser = argparse.ArgumentParser(description='Insula anatomy review API')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18082)
    parser.add_argument('--static', type=Path, default=None, help='Also serve this directory (production dist)')
    args = parser.parse_args()
    static_dir = args.static.resolve() if args.static else None
    if static_dir and not static_dir.is_dir():
        raise SystemExit(f'Static directory does not exist: {static_dir}')
    server = AnatomyServer((args.host, args.port), AnatomyHandler, static_dir)
    print(f'Anatomy API http://{args.host}:{args.port}', flush=True)
    if static_dir:
        print(f'Static files {static_dir}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping anatomy API', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
