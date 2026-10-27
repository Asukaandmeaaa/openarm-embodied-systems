"""No hardware access: all process checks, link reads and commands are mocked."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'preflight', Path(__file__).resolve().parents[1] / 'scripts/replay_can_preflight.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def ready():
    return dict(ifindex=22, flags=['UP'], mtu=72, txqlen=1000,
                linkinfo=dict(info_kind='can', info_data=dict(
                    ctrlmode=['FD'], state='ERROR-ACTIVE',
                    bittiming=dict(bitrate=1000000, sample_point='0.750'),
                    data_bittiming=dict(bitrate=5000000, sample_point='0.750', sjw=2))))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.links = {n: ready() for n in p.INTERFACES}
        self.guard = patch.object(p, 'require_no_hardware').start()
        self.read = patch.object(p, 'read_link', side_effect=lambda n: copy.deepcopy(self.links[n])).start()
        self.run = patch.object(p.subprocess, 'run', side_effect=self.command).start()
        self.addCleanup(patch.stopall)

    def command(self, args, **kwargs):
        self.assertNotIn('down', args)
        if args[-1] == 'up':
            self.links[args[5]] = ready()

    def test_ready_is_read_only(self):
        p.setup()
        self.run.assert_not_called()

    def test_down_interfaces_configured(self):
        for link in self.links.values():
            link['flags'] = []
        p.setup()
        self.assertEqual(self.run.call_count, 6)

    def test_missing_peer_prevents_all_writes(self):
        self.links['can0']['flags'] = []
        self.read.side_effect = [self.links['can0'], RuntimeError('missing can1')]
        with self.assertRaises(RuntimeError):
            p.setup()
        self.run.assert_not_called()

    def test_up_mismatch_prevents_all_writes(self):
        self.links['can0']['flags'] = []
        self.links['can1']['txqlen'] = 10
        with self.assertRaises(RuntimeError):
            p.setup()
        self.run.assert_not_called()

    def test_existing_hardware_prevents_reads_and_writes(self):
        self.guard.side_effect = RuntimeError('hardware running')
        with self.assertRaises(RuntimeError):
            p.setup()
        self.read.assert_not_called()
        self.run.assert_not_called()

    def test_command_failure_stops_setup(self):
        self.links['can0']['flags'] = []
        self.run.side_effect = subprocess.CalledProcessError(1, 'sudo')
        with self.assertRaises(subprocess.CalledProcessError):
            p.setup()
        self.assertEqual(self.run.call_count, 1)

    def test_bus_off_rejected(self):
        self.links['can0']['linkinfo']['info_data']['state'] = 'BUS-OFF'
        with self.assertRaises(RuntimeError):
            p.setup()
        self.run.assert_not_called()

    def test_reconnect_rejected(self):
        down = ready()
        down['flags'] = []
        changed = copy.deepcopy(down)
        changed['ifindex'] += 1
        self.read.side_effect = [down, ready(), changed]
        with self.assertRaises(RuntimeError):
            p.setup()
        self.run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
