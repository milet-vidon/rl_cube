"""Loopback UI for one operator stepping the live two-finger physics demonstration."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import secrets
from threading import Condition, Thread

PAGE = Path(__file__).with_name('natural_face90_demo.html')
HOST = re.compile(r'(?:127\.0\.0\.1|localhost)(?::[0-9]{1,5})?\Z', re.I)
PHASES = {'pickup', 'prepare', 'manual', 'stepping', 'replay', 'finished', 'failed'}


def valid_action(a):
    if not isinstance(a, dict):
        return False
    if set(a) == {'finish'}:
        return a['finish'] is True
    if set(a) - {'orbit'} != {'move', 'rotate', 'thumb', 'index', 'opposition', 'ticks'}:
        return False
    if type(a['ticks']) is not int or a['ticks'] not in (1, 5, 15):
        return False
    if any(type(a[k]) is not list or len(a[k]) != 3 for k in ('move', 'rotate')):
        return False
    orbit = a.get('orbit', 0)
    values = [*a['move'], *a['rotate'], a['thumb'], a['index'], a['opposition'], orbit]
    if any(type(v) is not int or abs(v) > 1 for v in values):
        return False
    return orbit == 0 or not (any(a['move']) or any(a['rotate']))


class DemoUI:
    """Enter before starting Kit. publish() accepts JSON-native values, one pose per body."""
    def __init__(self, port):
        self.port = port
        self._cv = Condition()
        self._session = secrets.token_hex(16)
        self._state = {'phase': 'starting'}
        self._revision = 0
        self._delivered = -1
        self._pending = None
        self._closed = False

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/'

    def __enter__(self):
        page = PAGE.read_bytes()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(4)

            def send_data(self, status, data, mime='application/json; charset=utf-8'):
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; "
                                 "style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; "
                                 "base-uri 'none'; form-action 'none'")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not HOST.fullmatch(self.headers.get('Host', '')):
                    self.send_data(403, b'{}')
                elif self.path == '/':
                    self.send_data(200, page, 'text/html; charset=utf-8')
                elif self.path == '/state':
                    with owner._cv:
                        rev = owner._revision
                        data = json.dumps(dict(owner._state, session=owner._session, revision=rev),
                                          ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
                    self.send_data(200, data)
                    with owner._cv:
                        owner._delivered = max(owner._delivered, rev)
                        owner._cv.notify_all()
                else:
                    self.send_data(404, b'{}')

            def do_POST(self):
                host = self.headers.get('Host', '')
                if self.path != '/action':
                    self.send_data(404, b'{}')
                    return
                if (not HOST.fullmatch(host) or self.headers.get('Origin') != 'http://' + host
                        or self.headers.get_content_type() != 'application/json'):
                    self.send_data(403, b'{}')
                    return
                try:
                    size = int(self.headers.get('Content-Length', ''))
                    if not 0 < size <= 1536:
                        raise ValueError('invalid request length')
                    req = json.loads(self.rfile.read(size))
                    if (not isinstance(req, dict) or set(req) != {'session', 'revision', 'action'}
                            or type(req['revision']) is not int or not valid_action(req['action'])):
                        raise ValueError('invalid command')
                except (ValueError, UnicodeError, RecursionError):
                    self.send_data(400, b'{}')
                    return
                with owner._cv:
                    if (owner._closed or req['session'] != owner._session or req['revision'] != owner._revision
                            or owner._state['phase'] != 'manual' or owner._pending is not None):
                        status = 409
                    else:
                        owner._state = dict(owner._state, phase='stepping')
                        owner._revision += 1
                        owner._pending = req['action']
                        owner._cv.notify_all()
                        status = 202
                self.send_data(status, b'{}')

            def log_message(self, fmt, *args):
                pass

        self._server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.port = self._server.server_port
        self._thread = Thread(target=self._server.serve_forever, kwargs={'poll_interval': .1}, daemon=True)
        self._thread.start()
        print(f'Demo UI {self.url}', flush=True)
        return self

    def publish(self, state):
        if state['phase'] not in PHASES:
            raise ValueError(f"Unknown demo phase: {state['phase']}")
        # A snapshot is independent of any tensors/lists mutated at the next physics tick.
        snapshot = json.loads(json.dumps(state, allow_nan=False))
        with self._cv:
            self._state = snapshot
            self._revision += 1
            self._cv.notify_all()

    def next_action(self):
        with self._cv:
            self._cv.wait_for(lambda: self._closed or self._pending is not None)
            if self._closed:
                raise RuntimeError('Demo UI closed')
            action, self._pending = self._pending, None
            return action

    def __exit__(self, *exc):
        with self._cv:
            if self._state['phase'] in ('finished', 'failed') and self._delivered >= 0:
                self._cv.wait_for(lambda: self._delivered >= self._revision, timeout=1.2)
            self._closed = True
            self._cv.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
