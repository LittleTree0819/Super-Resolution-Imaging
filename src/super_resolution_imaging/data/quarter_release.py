"""Non-training validation helpers for the frozen quarter-position release."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .writer import sha256_file


def load_legacy_acceptance(root,enabled,plan_path,cfg_path,dataset):
    """Explicit continuation policy changes photometry gating, never optics."""
    if not enabled:return None
    root=Path(root);plan_path=Path(plan_path);cfg_path=Path(cfg_path);dataset=Path(dataset)
    path=root/'configs/quarter_four_legacy_continuation.json'
    policy=json.loads(path.read_text(encoding='utf-8-sig'))
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    expected={'mode':'preserve_and_flag_known_photometry','user_authorized':True,
              'release_plan_sha256':sha256_file(plan_path),'config_sha256':sha256_file(cfg_path),
              'dataset':plan['proposed_new_dataset'],'original_manifest_sha256':plan['original_manifest_sha256'],
              'quality_policy_sha256':plan['quality_policy_sha256'],'brightness_threshold':.05,
              'no_frame_correction':True,'no_new_exclusions':True,'training_started':False}
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('Legacy continuation policy does not match frozen release')
    digest=sha256_file(path);copy=dataset/'legacy_continuation_policy.json'
    if copy.exists():
        if sha256_file(copy)!=digest:raise ValueError('Dataset continuation policy changed')
    else:shutil.copy2(path,copy)
    return digest


def apply_brightness_acceptance(check,policy_sha256):
    """Keep original threshold results visible even when explicitly accepted."""
    return {**check,'brightness_warning':bool(check['brightness_blocking']),
            'brightness_accepted_for_generation':bool(check['brightness_blocking'] and policy_sha256),
            'brightness_acceptance_policy_sha256':policy_sha256}


def atomic_json(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def image_mean(path,size):
    with Image.open(path) as image:
        if image.mode!='L' or image.size!=(size,size):raise ValueError(f'Invalid image: {path}')
        image.load()
        return float(np.asarray(image,dtype=np.float32).mean()/255.)


def verify_sample(folder,record,original_folder,original,positions,excluded):
    if record['sample_id']!=original['sample_id'] or record['split']!=original['split']:
        raise ValueError('Original ID or split changed')
    if record['positions_b_mm']!=positions:raise ValueError('Position order changed')
    if record['crop_box']!=original['crop_box'] or record['source_sha256']!=original['source_sha256']:
        raise ValueError('Original scene/crop provenance changed')
    if record['input_shape']!=[4,128,128] or record['target_shape']!=[1,256,256]:raise ValueError('Shape metadata changed')
    if sha256_file(folder/'source.png')!=sha256_file(original_folder/'source.png'):raise ValueError('Source crop changed')
    image_mean(folder/'source.png',768)
    if sha256_file(folder/'target.png')!=original['target_sha256'] or record['target_sha256']!=original['target_sha256']:
        raise ValueError('GT changed')
    image_mean(folder/'target.png',256)
    means=[]
    for group in ('a','b'):
        if len(record['output_sha256'][group])!=4:raise ValueError('Frame count mismatch')
        for frame in range(4):
            path=folder/group/f'frame_{frame:02d}.png'
            digest=sha256_file(path)
            if digest!=record['output_sha256'][group][frame]:raise ValueError(f'Hash mismatch: {path}')
            mean=image_mean(path,128)
            if group=='a' and digest!=original['output_sha256']['a'][frame]:raise ValueError('Center A changed')
            if group=='b':means.append(mean)
    center=image_mean(original_folder/'a/frame_00.png',128)
    mean_range=max(means)-min(means)
    return {'sample_id':record['sample_id'],'split':record['split'],'b_frame_means':means,
            'b_mean_range':mean_range,'max_b_center_mean_difference':max(abs(m-center) for m in means),
            'excluded_by_frozen_policy':record['sample_id'] in excluded,
            'brightness_blocking':mean_range>.05 and record['sample_id'] not in excluded}
