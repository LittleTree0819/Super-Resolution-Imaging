"""Reuse original center/GT/four frames and simulate only four new positions."""
import json
import shutil
from pathlib import Path
from super_resolution_imaging.config import load_config
from super_resolution_imaging.data.writer import DatasetWriter, write_json, sha256_file
from super_resolution_imaging.pipeline import json_safe, _normalize_grayscale, _verify_size
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import prepare_shift_model, set_lens_shift, validate_source
from super_resolution_imaging.zos.image_simulation import run_image_simulation
from dataclasses import asdict


def main():
    root=Path(__file__).resolve().parents[1]
    config=load_config(root/'configs/zemax_8frame_quarter_pilot.toml')
    plan_path=root/'data/four_layout_independent_128/selection.json'
    plan=json.loads(plan_path.read_text())
    if plan['count']!=128:raise ValueError('Unexpected plan')
    validate_source(config)
    geometry=json.loads((root/'runs/eight_frame_quarter_geometry/geometry.json').read_text())
    gate=json.loads((root/'runs/eight_frame_quarter_geometry/phase_gate.json').read_text())
    if not gate['passed'] or geometry['metadata']['source_model']['sha256']!=config.zemax.source_sha256:
        raise ValueError('Phase calibration is not valid for current model')
    if geometry['metadata']['positions_mm']!=[list(p) for p in config.dataset.positions_mm]:raise ValueError('Position calibration mismatch')
    writer=DatasetWriter(root/'runs/four_layout_independent_128',resume=True)
    snapshot={'config':json_safe(asdict(config)),'selection_sha256':sha256_file(plan_path),
              'generation':'reuse original four and GT; simulate new four only'}
    prepared=writer.root/'prepared_model.zmx'
    with connect(config.zemax.install_dir) as session:
        if writer.is_new:
            prepare_shift_model(session,config,prepared)
            snapshot['prepared_model_sha256']=sha256_file(prepared)
            write_json(writer.root/'dataset.json',snapshot)
        else:
            previous=json.loads((writer.root/'dataset.json').read_text())
            snapshot['prepared_model_sha256']=sha256_file(prepared)
            if previous!=snapshot:raise ValueError('Resume dataset configuration changed')
            session.system.LoadFile(str(prepared),False)
        completed=writer.completed_ids()
        for item in plan['samples']:
            sid=item['sample_id']
            if sid in completed:print('skip '+sid,flush=True);continue
            original=root/'runs/div2k_train_v1/samples'/item['original_sample_id']
            if sha256_file(original/'source.png')!=item['source_png_sha256']:raise ValueError('Original crop changed')
            if sha256_file(original/'target.png')!=item['target_sha256']:raise ValueError('Original target changed')
            if any(sha256_file(original/'b'/f'frame_{i:02d}.png')!=item['original_four_sha256'][i] for i in range(4)):
                raise ValueError('Original frames changed')
            folder=writer.root/'samples'/('.'+sid+'.partial')
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'a').mkdir(exist_ok=True);(folder/'b').mkdir(exist_ok=True)
            shutil.copy2(original/'source.png',folder/'source.png');shutil.copy2(original/'target.png',folder/'target.png')
            for i in range(8):shutil.copy2(original/'a/frame_00.png',folder/'a'/f'frame_{i:02d}.png')
            for i in range(4):shutil.copy2(original/'b'/f'frame_{i:02d}.png',folder/'b'/f'frame_{i:02d}.png')
            stage_name='sr_four_layout_128_'+sid+'.png'
            staged=config.zemax.images_dir/stage_name
            if staged.exists():raise FileExistsError('Staged image already exists; inspect before resuming: '+str(staged))
            shutil.copy2(folder/'source.png',staged)
            try:
                for i in range(4,8):
                    set_lens_shift(session,config,*config.dataset.positions_mm[i])
                    path=folder/'b'/f'frame_{i:02d}.png'
                    run_image_simulation(session,stage_name,path,config.simulation)
                    _normalize_grayscale(path);_verify_size(path,128)
            finally:staged.unlink(missing_ok=True)
            record={**item['original_record'],'sample_id':sid,'original_sample_id':item['original_sample_id'],
                'split':item['split'],'source_id':item['source_id'],'input_shape':[8,128,128],
                'positions_b_mm':[list(p) for p in config.dataset.positions_mm],
                'positions_a_mm':[[0.,0.]]*8,'source_png_sha256':item['source_png_sha256'],
                'output_sha256':{g:[sha256_file(folder/g/f'frame_{i:02d}.png') for i in range(8)] for g in ('a','b')}}
            write_json(folder/'metadata.json',record)
            destination=writer.root/'samples'/sid
            if destination.exists():raise FileExistsError(destination)
            folder.rename(destination);writer.append(record);completed.add(sid)
            print(f'completed {len(completed)}/128: {sid}',flush=True)
        writer.complete(len(completed))


if __name__=='__main__':main()
