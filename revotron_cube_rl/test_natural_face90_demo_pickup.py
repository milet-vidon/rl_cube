"""Exercise a complete operator handoff with CPU readback packets and no simulator."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from natural_face90_demo import DEFAULT_SEED, demonstrate
from test_natural_face90_demo_record import FakeEnv


class HandoffTests(unittest.TestCase):
    def run_episode(self, unsafe_frame=None, accept_step=29):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name)
        env = FakeEnv()
        cfg = json.loads(Path(__file__).with_name('config.json').read_bytes())
        env.names = cfg['joint_names']
        left = [i for i,n in enumerate(env.names) if n.startswith('left_') or n.endswith('_L')]
        env.cfg.update(cube_xyz=[.3,.4,1.], table_z=.8)
        env.cfg['success'].update(lift_m=.08, angular_speed_max=.1, linear_speed_max=.1, hold_seconds=10)
        env.domain_receipt['loss_grace_frames'] = 20
        env.home = torch.zeros(1,56)
        env.ids = list(range(56))
        q = env.home.clone()
        env.robot.get_dof_positions = lambda: q
        env.robot.get_dof_velocities = lambda: torch.zeros_like(q)
        env.core = lambda: env.views['Core'].get_transforms().float()
        links = ['world','Link7_L','Link7_R','right_thumb_DIP_link','right_index_DIP_link']
        env.robot.shared_metatype = SimpleNamespace(link_names=links)
        link_poses = env.core()[:,None].repeat(1,len(links),1)
        link_poses[0,:,0] += torch.arange(len(links))*.01
        env.robot.get_link_transforms = lambda: link_poses
        env.limits = torch.stack((q-2,q+2),-1)
        env.hard, env.step, env.vel = env.limits, q+.02, q+5
        env.hand_impulse = torch.tensor([[.02,0.]])
        env.offset = torch.zeros(1,3)
        env.feature_contacts = torch.zeros(1,10)
        env.views['Core'].get_velocities = lambda: torch.zeros(1,6)
        env.batch_gain = Mock()
        env.reset = Mock(side_effect=lambda seeds: env.contacts[:,12].fill_(.01))

        true = torch.tensor([True])
        false = ~true
        commands = []
        def physics(scene, command, output):
            commands.append(command.clone())
            q.copy_(command)
            scene.contacts.zero_()
            if len(commands)-1 == unsafe_frame:
                scene.contacts[:,10] = .02
            scene.frame([[]]*8)

        def teach(active, joint, dq, core, target, feedback, velocity, impulse):
            return (target+.002, torch.tensor([0]), torch.tensor([0]),
                    torch.tensor([len(commands)==accept_step]))
        teacher = SimpleNamespace(action=teach)
        model = Mock()
        model.to.return_value.eval.return_value = model
        model.side_effect = lambda obs: obs[:,112:168][:,left]+.001
        observer = Mock()
        observer.observe.return_value = dict(natural_ready=true, structure_ok=true,
                                             allowed_layer_state=true, wrong_contact=false)
        prepare = Mock(detour_done=true, phase=torch.tensor([2]))
        prepare.action.side_effect = lambda target, *args: (target+.003, false, false, [''])
        manual = Mock(placo=SimpleNamespace(receipt={}))
        manual.step.side_effect = lambda target, action: target.clone()
        packet = dict(move=[0,0,0], rotate=[0,0,0], orbit=1, thumb=1, index=1, opposition=0, ticks=5)
        ui = Mock()
        ui.next_action.side_effect = [packet, {'finish':True}]
        with ExitStack() as stack:
            for path, result in {
                'natural_face90_demo_scene.open_demo_scene':env,
                'contact_hold90_cuda_domain.make_plan':SimpleNamespace(accepted=true),
                'contact_hold90_cuda_domain.make_teacher':teacher,
                'contact_hold90_cuda_sample.local_palm_reader':lambda core:core[:,:3],
                'natural_face90_v5_metrics.NaturalFace90Observer':observer,
                'natural_face90_v5_controller.NaturalFace90Controller':prepare,
                'natural_face90_demo_control.ManualControl':manual,
            }.items():
                stack.enter_context(patch(path,return_value=result))
            stack.enter_context(patch('natural_face90_recipe_contacts.step_physics',side_effect=physics))
            result = demonstrate(ui,DEFAULT_SEED,out,('source','grasp',
                dict(output_indices=left, joint_names=env.names, sources=[dict(candidate={})]),model))
        env.reset.assert_called_once_with([DEFAULT_SEED])
        manual.close.assert_called_once_with()
        self.assertEqual(result, json.loads((out/'demo_summary.json').read_text()))
        rows = [json.loads(line) for line in (out/'demo_frames.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),len(commands))
        return result, rows, prepare, ui

    def test_operator_receives_the_last_preparation_command_in_continuous_episode(self):
        result,rows,prepare,ui = self.run_episode()
        self.assertEqual([r['phase'] for r in rows],['pickup']*30+['prepare']+['manual']*5)
        self.assertEqual(rows[31]['command_rad'], rows[30]['command_rad'])
        self.assertEqual(rows[31]['q_before_rad'],rows[30]['q_rad'])
        self.assertTrue(result['handoff_recorded'])
        self.assertEqual(result['unsafe_frames'],0)
        prepare.begin.assert_called_once()
        self.assertEqual(ui.next_action.call_count,2)
        for call in ui.publish.call_args_list:
            snapshot = call.args[0]
            frame = rows[snapshot['step']]
            self.assertEqual(snapshot['bodies'],frame['bodies'])
            self.assertEqual(snapshot['thumb_index_same_substeps'],frame['thumb_index_same_substeps'])
        self.assertIn('right_thumb_DIP_link',rows[31]['bodies'])
        self.assertNotEqual(rows[31]['bodies']['right_thumb_DIP_link'],rows[31]['bodies']['right_index_DIP_link'])
        self.assertTrue(all(row['action']['orbit']==1 for row in rows[31:]))

    def test_early_teacher_acceptance_does_not_bypass_thirty_frame_hold(self):
        result,rows,prepare,ui = self.run_episode(accept_step=28)
        self.assertEqual(len(rows),29)
        self.assertFalse(result['handoff_recorded'])
        self.assertEqual(result['status'],'failed')
        prepare.begin.assert_not_called()
        ui.next_action.assert_not_called()

    def test_collision_stops_a_multi_tick_command_on_its_first_unsafe_frame(self):
        result,rows,_,ui = self.run_episode(unsafe_frame=32)
        self.assertEqual(len(rows),33)
        self.assertEqual(result['manual_frames'],2)
        self.assertEqual(result['unsafe_frames'],1)
        self.assertEqual(result['reason'],'robot_self_contact')
        ui.next_action.assert_called_once()


if __name__ == '__main__':
    unittest.main()
