import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import contact_hold_cuda_train as campaign
from contact_hold_cuda_monitor import first_error
from contact_hold_loop_data import read,write

class Recovery(unittest.TestCase):
    def test_original_oom_and_corruption_are_different(self):
        allocator='[Error] [omni.physx.plugin] PhysX error: PxgCudaDeviceMemoryAllocator failed to allocate memory 415576064 bytes! Result = 2'
        self.assertEqual(first_error(allocator)[0],'memory_pressure')
        self.assertEqual(first_error('[Error] [omni.physx.plugin] PhysX error: Scene state is corrupted')[0],'native_error')
        self.assertIsNone(first_error('GPU_COLLECTION_COMPLETE 1536 lanes'))

    def test_retry_full_seed_range_and_resume_with_smaller_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd=Path(tmp);st=rd/'state.json';s={};calls=[]
            def simulate(cmd,log,fds):
                dst=Path(cmd[cmd.index('--output')+1]);p=read(dst.parent/'collect.json')
                off,n=p['lane_offset'],len(p['seeds']);calls.append((off,n))
                if (off,n) in ((0,16),(8,8)):return dict(ok=False,kind='memory_pressure',reason='allocation failed')
                dst.mkdir();write(dst/'report.json',{'num_envs':n,'seeds':p['seeds'],'checkpoint':'ref','samples':n,'ends':['test']*n})
                return dict(ok=True,kind=None,reason=None)
            from contextlib import nullcontext
            with patch.object(campaign,'monitor',simulate),patch.object(campaign,'checkpoint',lambda _: 'ref'),patch.object(campaign,'sampler_code',lambda:{}),patch.object(campaign.loop,'gpu_slot',lambda _:nullcontext(())),patch.object(campaign,'verify_wave',lambda e:({},read(Path(e['path'])/'report.json'))),patch.object(campaign,'verify_batch',lambda _:None):
                kw=(rd,2,rd,{'envs':16},s,st)
                bundle=campaign.collect(*kw)
                self.assertEqual(calls,[(0,16),(0,8),(8,8),(8,4),(12,4)])
                self.assertEqual(campaign.collect(*kw),bundle)
                self.assertEqual(len(calls),5)
                seeds=[v for w in bundle['waves'] for v in read(Path(w['path'])/'report.json')['seeds']]
                self.assertEqual(seeds,list(range(1004096,1004112)))

    def test_signalled_worker_can_restart(self):
        import tempfile
        from contextlib import nullcontext
        from pathlib import Path
        from unittest.mock import patch
        from contact_hold_loop_data import read,write
        with tempfile.TemporaryDirectory() as tmp:
            rd=Path(tmp);state=rd/'state.json';s={};calls=[]
            def simulate(cmd,log,fds):
                dst=Path(cmd[cmd.index('--output')+1]);p=read(dst.parent/'collect.json');calls.append(dst)
                if len(calls)==1:return dict(ok=False,kind='worker_exit',reason='worker returned -15',exit_code=-15)
                dst.mkdir();write(dst/'report.json',dict(num_envs=4,seeds=p['seeds'],checkpoint='ref',samples=4,ends=['test']*4))
                return dict(ok=True,kind=None,reason=None,exit_code=0)
            with patch.object(campaign,'monitor',simulate),patch.object(campaign,'sampler_code',lambda:{}),patch.object(campaign,'checkpoint',lambda _: 'ref'),patch.object(campaign.loop,'gpu_slot',lambda _:nullcontext(())),patch.object(campaign,'verify_wave',lambda e:({},read(Path(e['path'])/'report.json'))),patch.object(campaign,'verify_batch',lambda _:None):
                kw=(rd,2,rd,{'envs':4},s,state)
                with self.assertRaisesRegex(RuntimeError,'-15'):campaign.collect(*kw)
                out=campaign.collect(*kw)
                self.assertEqual(len(calls),2);self.assertNotEqual(calls[0],calls[1])
                self.assertEqual(read(calls[0].with_suffix('.resources.json'))['exit_code'],-15)
                self.assertEqual(out['waves'][0]['path'],str(calls[1]))

if __name__=='__main__':unittest.main()
