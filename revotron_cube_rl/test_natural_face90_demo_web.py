"""Protocol tests with synthetic UI frames, no Kit, PhysX or robot drivers."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import unittest

from natural_face90_demo_web import DemoUI, valid_action


def move():
    return dict(move=[1, 0, 0], rotate=[0, 0, 0], index=0, thumb=0, opposition=0, ticks=1)


class DemoWebTests(unittest.TestCase):
    def setUp(self):
        self.ui = self.enterContext(DemoUI(0))

    def request(self, method, path, data=None, **headers):
        c = HTTPConnection('127.0.0.1', self.ui.port, timeout=2)
        try:
            c.request(method, path, json.dumps(data).encode() if data is not None else None, headers)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def command(self, state, action=None, **headers):
        return self.request('POST', '/action', dict(session=state['session'], revision=state['revision'],
                            action=move() if action is None else action),
                            **{'Origin': self.ui.url.rstrip('/'), 'Content-Type':'application/json', **headers})[0]

    def state(self):
        status, data = self.request('GET', '/state')
        self.assertEqual(status, 200)
        return json.loads(data)

    def test_single_command_per_observed_manual_frame_and_reconnect(self):
        self.assertEqual(self.request('GET', '/')[0], 200)
        for phase in ('prepare','pickup','replay'):
            self.ui.publish(dict(phase=phase))
            self.assertEqual(self.command(self.state()), 409)
        self.ui.publish(dict(phase='manual', step=1))
        state = self.state()
        with ThreadPoolExecutor(max_workers=4) as pool:
            codes = list(pool.map(lambda _: self.command(state), range(4)))
        self.assertCountEqual(codes, [202, 409, 409, 409])
        self.assertEqual(self.ui.next_action(), move())
        self.assertEqual(self.state()['phase'], 'stepping')
        self.ui.publish(dict(phase='manual', step=2))
        self.assertEqual(self.command(state), 409)
        self.assertEqual(self.command(self.state(),{'finish':True}), 202)
        self.assertEqual(self.ui.next_action(), {'finish':True})
        self.ui.publish(dict(phase='finished'))
        self.assertEqual(self.command(self.state()), 409)

    def test_same_origin_and_bounded_input(self):
        self.ui.publish(dict(phase='manual'))
        s = self.state()
        self.assertEqual(self.request('GET','/state', Host='invalid.example')[0],403)
        self.assertEqual(self.command(s, Origin='http://unrelated.example'),403)
        self.assertEqual(self.command(s, **{'Content-Type':'text/plain'}),403)
        for a in [dict(move(), ticks=0), dict(move(), move=[1,2,0]),dict(move(), index=True),
                  dict(move(), ticks=1.0), dict(move(), payload=1), {'finish':1}]:
            self.assertFalse(valid_action(a))
            self.assertEqual(self.command(s,a),400)
        # Forwarding e.g. local port 8890 to a different remote port is supported.
        self.assertEqual(self.command(s, Host='localhost:8890', Origin='http://localhost:8890'),202)

    def test_face_axis_orbit_with_wrist_at_rest(self):
        for direction in (-1, 1):
            self.ui.publish(dict(phase='manual'))
            state = self.state()
            orbit = dict(move(), move=[0, 0, 0], orbit=direction, ticks=5)
            for invalid in [dict(orbit, move=[0, 0, 1]), dict(orbit, rotate=[-1, 0, 0]),
                            dict(orbit, orbit=True), dict(orbit, orbit=.5), dict(orbit, orbit=2)]:
                self.assertEqual(self.command(state, invalid), 400)
            self.assertEqual(self.command(state, orbit), 202)
            self.assertEqual(self.ui.next_action(), orbit)
        self.ui.publish(dict(phase='manual'))
        zero_orbit = dict(move(), orbit=0)
        self.assertEqual(self.command(self.state(), zero_orbit), 202)
        self.assertEqual(self.ui.next_action(), zero_orbit)


if __name__ == '__main__':
    unittest.main()
