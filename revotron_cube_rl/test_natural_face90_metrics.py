import unittest,math
import torch
from natural_face90_metrics import Layer,Sequence,actor_class,mul,rotate

def states():
    parts=['_'.join(x for x in (a,b,c) if x) for a in ('','pX','nX') for b in ('','pY','nY') for c in ('','pZ','nZ') if a or b or c];core=torch.zeros((2,7),dtype=torch.float64);core[:,6]=1;p=core[:,None].repeat(1,26,1)
    for i,n in enumerate(parts):
        for j,axis in enumerate('XYZ'):
            for sgn,pref in [(1,'p'),(-1,'n')]:
                if pref+axis in n.split('_'):p[:,i,j]=sgn*.03
    return parts,core,p

def quat(deg,axis):
    r=torch.zeros(4,dtype=torch.float64);r[axis]=math.sin(math.radians(deg)/2);r[3]=math.cos(math.radians(deg)/2);return r

class PhysicsGates(unittest.TestCase):
    def test_legal_orbit_and_global_rotation(self):
        parts,c,p=states();metric=Layer(parts,c,p);sel=metric.select;r=quat(-15,2);pc=p.clone();pc[:,sel,:3]=rotate(r,pc[:,sel,:3]);pc[:,sel,3:]=mul(r,pc[:,sel,3:]);w=quat(103,0);c[:,3:]=w;c[:,:3]=torch.tensor([1,-2,3]);pc[:,:,:3]=rotate(w,pc[:,:,:3])+c[:,None,:3];pc[:,:,3:]=mul(w,pc[:,:,3:]);pc[1,::2,3:]*=-1
        m=metric.measure(c,pc,15);self.assertTrue(m['at_goal'].all());self.assertTrue(torch.allclose(m['face_deg'],c.new_tensor([15,15])));self.assertLess(m['anchor_error_m'].max(),1e-9)
        pc[0,sel.nonzero()[0,0],0]+=.004;self.assertFalse(metric.measure(c,pc,15)['structure_ok'][0]);pc[1,(~sel).nonzero()[0,0],3:]=mul(w,quat(18,2));self.assertFalse(metric.measure(c,pc,15)['structure_ok'][1]);
    def test_whole_only_wrong_direction_or_off_axis(self):
        parts,c,p=states();metric=Layer(parts,c,p);w=quat(95,1);c[:,3:]=w;p[:,:,:3]=rotate(w,p[:,:,:3]);p[:,:,3:]=w;m=metric.measure(c,p,15);self.assertTrue(m['assembled'].all());self.assertFalse(m['at_goal'].any());parts,c,p=states();metric=Layer(parts,c,p);sel=metric.select
        p[0,sel,3:]=quat(15,2);p[0,sel,:3]=rotate(quat(15,2),p[0,sel,:3]);p[1,sel,3:]=quat(15,0);m=metric.measure(c,p,15);self.assertLess(m['face_deg'][0],0);self.assertFalse(m['at_goal'].any());self.assertFalse(m['structure_ok'][1]);
    def test_actual_source_paths(self):
        self.assertEqual(actor_class('/World/envs/env_0012/Robot/right_index_DIP_link'),(12,1));self.assertEqual(actor_class('/World/envs/env_0003/Robot/Link6_R'),(3,2));self.assertEqual(actor_class('/World/envs/env_0001/Robot/left_ring_PIP_link'),(1,3));self.assertEqual(actor_class('/World/envs/env_0001/Cube/nZ'),(1,0));
        with self.assertRaises(ValueError):actor_class('/environment/right_index_dummy')
    def test_order_direct_drive_true_retreat_and_hold(self):
        seq=Sequence(2,'cpu');b=lambda a:torch.tensor(a,dtype=torch.bool);d=lambda a:torch.tensor(a,dtype=torch.float64);yes=b([1,1]);no=~yes;rel=torch.zeros(2,3,dtype=torch.float64)
        def run(ang,touch,retract=False,dist=.08,nat=True):
            a=d(ang);g=dict(face_deg=a,structure_ok=yes,assembled=a.abs()<=5,at_goal=(a-15).abs()<=5);contact=b(touch);return seq.update(yes,yes,yes&nat,yes,contact,no,contact,d([dist]*2),yes&retract,yes,yes,g,rel,d([1,1]),yes.long()*30,yes.long()*90)
        for _ in range(29):o=run([0,0],[0,0]);self.assertFalse(o['natural_ready'].any())
        self.assertTrue(run([0,0],[0,0])['natural_ready'].all());run([0,0],[1,1]);run([5,0],[1,0]);run([11,13],[1,0]);run([15,15],[1,1]);o=run([15,15],[0,0],True,.08);self.assertFalse(o['hold_fraction'].any());self.assertTrue(o['turn_started'][0]);self.assertFalse(o['turn_started'][1]);
        for _ in range(89):o=run([15,15],[0,0],True,.11);self.assertFalse(o['success'].any())
        o=run([15,15],[0,0],True,.11);self.assertEqual(o['success'].tolist(),[True,False])

    def test_12deg_does_not_qualify_and_withdrawal_may_only_lose_two_degrees(self):
        seq=Sequence(3,'cpu');yes=torch.ones(3,dtype=torch.bool);no=~yes;d=lambda x:torch.tensor(x,dtype=torch.float64);rel=torch.zeros(3,3,dtype=torch.float64)
        def frame(angles,touch=False,withdraw=False):
            a=d(angles);g=dict(face_deg=a,structure_ok=yes,assembled=a.abs()<=5,at_goal=(a-15).abs()<=5);return seq.update(yes,yes,yes,yes,yes&touch,no,yes&touch,d(([.12]*3) if withdraw else ([.08]*3)),yes&withdraw,yes,yes,g,rel,d([1]*3),yes.long()*30,yes.long()*90)
        for _ in range(30):frame([0]*3)
        frame([0]*3,True);frame([12,15,16],True)
        for _ in range(95):m=frame([12,13,13],withdraw=True)
        self.assertEqual(m['contact_goal_reached'].tolist(),[False,True,True]);self.assertEqual(m['success'].tolist(),[False,True,False]);self.assertEqual(m['retreat_floor_deg'].tolist(),[13,13,14])

class RawPairReadback(unittest.TestCase):
    def test_same_force_on_other_layer_or_other_hand_is_not_right_digit_target(self):
        # PhysX packet offsets are not contiguous in sensor order. Exercise real
        # reducer unpacking, rather than handing the phase machine labeled contacts.
        from types import SimpleNamespace
        from natural_face90_metrics import TargetContacts
        names=states()[0];paths=['/World/envs/env_0000/Cube/'+x for x in names]
        count=torch.zeros(26,dtype=torch.int32);start=count.clone();count[names.index('nZ')]=2;start[names.index('nZ')]=3;count[names.index('pY')]=2;start[names.index('pY')]=0
        force=torch.tensor([.81,.32,0,.05,.4,0],dtype=torch.float32);other=torch.tensor([11,12,11,11,12,12],dtype=torch.int64)
        v=SimpleNamespace(sensor_paths=paths,sensor_count=26,max_contact_data_count=20,get_raw_contact_data=lambda dt:(force,None,None,None,count,start,other));base=SimpleNamespace(update=lambda s:None,clear=lambda s:None,validate=lambda:None,cache={11:(),12:()});scene=SimpleNamespace(n=1,home=torch.zeros(1),reducer=base,contact_views=[('cube',v,[0]*26)])
        obs=TargetContacts(scene,names);obs.cache={11:(0,1),12:(0,3)};obs.keys=torch.tensor([11,12]);obs.code=torch.tensor([[0,1],[0,3]]);obs.clear(scene);obs.update(scene);assert torch.allclose(obs.acc[0],torch.tensor([.05,.81,.86,.32,.4],dtype=torch.float64));obs.update(scene);self.assertAlmostEqual(obs.acc[0,0].item(),.1,places=6)
        obs.code[0,0]=1
        with self.assertRaisesRegex(ValueError,'Cross-lane'):obs.update(scene)

class MeasuredSteadiness(unittest.TestCase):
    def test_stationary_poses_different_frame_counts_and_signs(self):
        from natural_face90_metrics import PoseWindows
        w=PoseWindows(2,'cpu');_,core,p=states();pieces=p[:,:9];active=torch.ones(2,dtype=torch.bool)
        for k in range(90):
            q=pieces[:,:,3:] * (-1 if k%2 else 1);c=core.clone();c[:,3:]*=(-1 if k%2 else 1)
            m=w.measure(active,c,pieces[:,:,:3],q);self.assertEqual(m['natural_motion_ok'].tolist(),[k>=29]*2);self.assertEqual(m['retreat_motion_ok'].tolist(),[k>=89]*2)
        self.assertLess(m['retreat_piece_span_deg'].max(),1.e-6);active[1]=False;m=w.measure(active,core,pieces[:,:,:3],pieces[:,:,3:]);self.assertEqual(m['retreat_steady_suffix_frames'].tolist(),[90,0]);
    def test_drift_or_single_piece_excursion_with_same_median_rejected(self):
        from natural_face90_metrics import PoseWindows
        w=PoseWindows(2,'cpu');_,core,p=states();pieces=p[:,:9];active=torch.ones(2,dtype=torch.bool)
        for k in range(90):
            # 0.4 mm / control frame looks slow but travels 35.6 mm over the hold.
            c=core.clone();c[0,0]=k*.0004;q=pieces[:,:,3:].clone();q[1,0]=quat(3.5 if k==65 else 0,0);m=w.measure(active,c,pieces[:,:,:3],q)
        self.assertAlmostEqual(m['retreat_core_span_m'][0].item(),.0356,places=5);self.assertAlmostEqual(m['retreat_piece_span_deg'][1].item(),3.5,places=4);self.assertEqual(m['natural_motion_ok'].tolist(),[False,False]);self.assertEqual(m['retreat_motion_ok'].tolist(),[False,False]);self.assertEqual(m['retreat_steady_suffix_frames'][1].item(),24)
        for k in range(6):m=w.measure(active,c,pieces[:,:,:3],pieces[:,:,3:]);
        self.assertTrue(m['natural_motion_ok'][1]);self.assertFalse(m['retreat_motion_ok'][1])
    def test_core_rotation_and_relative_translation_independent(self):
        from natural_face90_metrics import PoseWindows
        w=PoseWindows(2,'cpu');_,core,p=states();pieces=p[:,:9];active=torch.ones(2,dtype=torch.bool)
        for k in range(30):
            c=core.clone();pp=pieces[:,:,:3].clone();c[0,3:]=quat(8*k/29,1);pp[1,5,1]+=k*.003/29;m=w.measure(active,c,pp,pieces[:,:,3:]);
        self.assertEqual(m['natural_motion_ok'].tolist(),[False,False]);self.assertAlmostEqual(m['natural_core_span_deg'][0].item(),8.,places=3);self.assertAlmostEqual(m['natural_piece_span_m'][1].item(),.003,places=5)
