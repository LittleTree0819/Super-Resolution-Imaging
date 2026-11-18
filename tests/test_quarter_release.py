"""Check that the new release audit rejects changed GT and flags photometric drift."""
import tempfile
import json
import unittest
from pathlib import Path
from PIL import Image
from super_resolution_imaging.data.writer import sha256_file
from super_resolution_imaging.data.quarter_release import verify_sample,apply_brightness_acceptance,load_legacy_acceptance


class QuarterReleaseTests(unittest.TestCase):
    def test_preservation_and_brightness_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            original=Path(temp)/'old';folder=Path(temp)/'new'
            for root in (original,folder):
                for group in ('a','b'):(root/group).mkdir(parents=True)
                Image.new('L',(768,768),120).save(root/'source.png')
                Image.new('L',(256,256),120).save(root/'target.png')
                for i in range(4):
                    Image.new('L',(128,128),120).save(root/'a'/f'frame_{i:02d}.png')
                    Image.new('L',(128,128),120+30*(i==3)).save(root/'b'/f'frame_{i:02d}.png')
            positions=[[.1,.1],[-.1,.1],[-.1,-.1],[.1,-.1]]
            record={'sample_id':'sample','split':'train','source_sha256':'same','crop_box':[0,0,768,768],
                'input_shape':[4,128,128],'target_shape':[1,256,256],'positions_b_mm':positions,
                'target_sha256':sha256_file(original/'target.png'),
                'output_sha256':{g:[sha256_file(folder/g/f'frame_{i:02d}.png') for i in range(4)] for g in ('a','b')}}
            result=verify_sample(folder,record,original,record,positions,[])
            self.assertTrue(result['brightness_blocking'])
            accepted=apply_brightness_acceptance(result,'explicit-policy-sha')
            self.assertTrue(accepted['brightness_blocking'])
            self.assertTrue(accepted['brightness_accepted_for_generation'])
            self.assertFalse(apply_brightness_acceptance(result,None)['brightness_accepted_for_generation'])
            self.assertFalse(verify_sample(folder,record,original,record,positions,['sample'])['brightness_blocking'])
            Image.new('L',(256,256),119).save(folder/'target.png')
            with self.assertRaisesRegex(ValueError,'GT changed'):
                verify_sample(folder,record,original,record,positions,[])

    def test_legacy_acceptance_bound_to_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'configs').mkdir();dataset=root/'dataset';dataset.mkdir()
            cfg=root/'configs/frozen.toml';cfg.write_text('frozen optics')
            plan_path=root/'configs/plan.json'
            plan={'proposed_new_dataset':'dataset','original_manifest_sha256':'source-manifest','quality_policy_sha256':'exclusions'}
            plan_path.write_text(json.dumps(plan))
            path=root/'configs/quarter_four_legacy_continuation.json'
            policy={'mode':'preserve_and_flag_known_photometry','user_authorized':True,
                    'release_plan_sha256':sha256_file(plan_path),'config_sha256':sha256_file(cfg),
                    'dataset':'dataset','original_manifest_sha256':'source-manifest','quality_policy_sha256':'exclusions',
                    'brightness_threshold':.05,'no_frame_correction':True,'no_new_exclusions':True,'training_started':False}
            path.write_text(json.dumps(policy))
            self.assertIsNone(load_legacy_acceptance(root,False,plan_path,cfg,dataset))
            digest=load_legacy_acceptance(root,True,plan_path,cfg,dataset)
            self.assertEqual(digest,sha256_file(dataset/'legacy_continuation_policy.json'))
            cfg.write_text('changed optics')
            with self.assertRaisesRegex(ValueError,'does not match'):
                load_legacy_acceptance(root,True,plan_path,cfg,dataset)
