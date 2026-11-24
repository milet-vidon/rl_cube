"""Exercise the live worker handoff, contacts, Placo, PPO and evidence files with CPU sensor packets."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import natural_face90_bimanual as campaign
from natural_face90_bimanual_policy import ACTOR_DIM, ACTION_DIM
from natural_face90_bimanual_worker import BimanualPhase, LaneEvidence
from natural_face90_placo_client import self_check
from kinematics import RobotKinematics
from test_natural_face90_bimanual_worker import Packets


class Readback:
    """A solver-independent readback fixture with a stationary supported layer."""
    def __init__(self, raw):
        self.r = raw
        self.machine = SimpleNamespace(target=15.)
        self.last_step = -1
        self.ready = False

    def observe(self, active, post, safe, retract, step):
        self.last_step = step
        f = torch.zeros_like(active)
        yes = ~f
        z = torch.zeros_like(active, dtype=torch.float64)
        a = self.r.acc
        return dict(natural_ready=yes & self.ready, turn_started=f, success=f, support=yes,
                    structure_ok=yes, allowed_layer_state=yes, wrong_contact=f, progress=z,
                    posture_quality=z+1, right_contact=a[:,0]>0, hold_fraction=z, face_deg=z,
                    target_impulse=a[:,0], other_impulse=a[:,1], left_static_impulse=a[:,3],
                    left_target_impulse=a[:,4], contact_goal_reached=f)


class BimanualHandoff(unittest.TestCase):
    def test_handoff_two_lanes_real_ppo_and_replayable_evidence(self):
        here = Path(__file__).parent
        cfg = json.loads((here/'config.json').read_bytes())
        names = cfg['joint_names']
        preset = json.loads((here/'natural_face90_v5_regression.json').read_bytes())['r27_endpoint']['q']
        q = torch.tensor([preset, preset], dtype=torch.float64)
        limits = RobotKinematics().limits
        bounds = q.new_tensor([[limits[n] for n in names]])
        q = q.clamp(bounds[:,:,0], bounds[:,:,1])
        root = q.new_tensor([[0.,0.,cfg['robot_z'],0.,0.,0.,1.]]).expand(2,-1)
        core = root.clone()
        core[:, :3] += q.new_tensor([.56, .25, .20])
        link_names = ('world','Link4_R','Link5_R','Link7_R','right_index_DIP_link')
        paths = [[f'/World/envs/env_{i:04}/Robot/{n}' for n in link_names] for i in range(2)]
        robot = SimpleNamespace(link_paths=paths, get_dof_positions=lambda:q, get_root_transforms=lambda:root)
        e = SimpleNamespace(n=2,names=names,ids=list(range(56)),cfg=cfg,robot=robot,home=q,dt=1/240,dec=8,
                            limits=bounds,step=q.new_full((1,56),.025),vel=q.new_full((1,56),1.57),
                            domain_receipt={'scene_sha256':'fixture'},views={'Core':SimpleNamespace(get_transforms=lambda:core)})
        raw = Packets()
        monitor = Readback(raw)
        def clear(_e):
            raw.acc.zero_()
            raw.count = 0
        pairs = SimpleNamespace(base=raw, clear=clear, update=lambda _:None, validate=lambda:None)
        req = campaign.request(dict(rng=327,source_sha256=campaign.fingerprint(),placo=self_check()['receipt']),
                               'train',campaign.block(campaign.TRAIN_BASE,1,2),1,2,None,None)
        def paths_for_actors(scene, ids):
            digits = ['thumb','index','middle']
            return [f'/World/envs/env_{a//3:04}/Robot/right_{digits[a%3]}_DIP_link' for a in ids]

        with TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch('natural_face90_v5_metrics.NaturalFace90Observer', return_value=monitor))
            stack.enter_context(patch('natural_face90_v5_contacts.PairRecorder', return_value=pairs))
            stack.enter_context(patch('natural_face90_bimanual_contacts._actor_paths', side_effect=paths_for_actors))
            folder = Path(tmp)
            phase = BimanualPhase(e, req, folder, stack)
            for stream in (phase.evidence, phase.traces, phase.incidents):
                stack.callback(stream.close)
            self.assertEqual((monitor.target, monitor.machine.target), (2., 2.))
            self.assertIs(e.reducer, phase.tap)
            active = torch.ones(2, dtype=torch.bool)
            steps = torch.zeros(2, dtype=torch.long)
            target = q.clone()

            def substeps(loaded):
                phase.tap.clear(e)
                for _ in range(8):
                    # First lane grips opposing material columns, second lane touches only with the thumb.
                    raw.emit([(0,0,.001),(4,1,.001),(26,3,.001)] if loaded else [])
                    raw.acc[:,3] += .04
                    phase.tap.update(e)
                phase.tap.validate()

            substeps(False)
            phase.observe(active,active,active,0)
            phase.propose(target,active,steps)
            self.assertIsNone(phase.pending)
            monitor.ready = True
            phase.controller.phase[:] = 2
            phase.controller.detour_done[:] = True
            phase.observe(active,active,active,1)
            for k in range(1,17):
                cmd = phase.propose(target,active,steps)
                self.assertEqual(phase.pending['obs'].shape,(2,ACTOR_DIM))
                self.assertEqual(phase.pending['latent'].shape,(2,ACTION_DIM))
                torch.testing.assert_close(cmd[:,phase.left_ids], phase.hold_target[:,phase.left_ids])
                q.copy_(cmd)
                target = cmd
                substeps(True)
                result = phase.observe(active,active,active,k+1)
                self.assertFalse(result['success'].any())
                steps += 1
                alive = active if k<16 else ~active
                phase.after_step(active,alive,~active,['post_horizon']*2,steps)
            info = phase.finish_learning()
            self.assertEqual(info['learned_transitions'],32)
            self.assertTrue(info['actor_updated'])
            self.assertTrue((folder/'checkpoint.pt').is_file())
            self.assertEqual(phase.counts['bilateral_frames'].tolist(),[16,0])
            self.assertEqual(phase.grip.peak.tolist(),[0.,0.])
            evidence = [json.loads(s) for s in (folder/'bimanual_contacts.jsonl').read_text().splitlines()]
            self.assertEqual(len(evidence),32)
            baseline = {(m['step'],m['lane']):m for m in
                        map(json.loads,(folder/'post_measurements.jsonl').read_text().splitlines())}
            lanes = [LaneEvidence(),LaneEvidence()]
            for r in evidence:
                lanes[r['lane']].read(r,baseline[r['step'],r['lane']],2)
            for lane, checked in enumerate(lanes):
                checked.check(dict(phase.row(lane),unsafe=False,success=False),2)


if __name__=='__main__':
    unittest.main()
