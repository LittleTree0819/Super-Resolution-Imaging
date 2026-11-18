"""Generate the frozen four-position dataset; no training code is invoked."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from super_resolution_imaging.config import load_config
from super_resolution_imaging.pipeline import json_safe, _normalize_grayscale, _verify_size
from super_resolution_imaging.data.writer import DatasetWriter, sha256_file
from super_resolution_imaging.data.quarter_release import atomic_json,verify_sample,load_legacy_acceptance,apply_brightness_acceptance
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import validate_source,prepare_shift_model,set_lens_shift
from super_resolution_imaging.zos.image_simulation import run_image_simulation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('canary','train-validation','test'),required=True)
    parser.add_argument('--accept-known-photometry',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    plan_path=root/'configs/quarter_four_release_plan.json';plan=json.loads(plan_path.read_text())
    cfg_path=Path(plan['new_config']);cfg=load_config(cfg_path);old=Path(plan['original_dataset'])
    if sha256_file(cfg_path)!=plan['new_config_sha256'] or sha256_file(old/'manifest.jsonl')!=plan['original_manifest_sha256']:
        raise ValueError('Frozen config or source manifest changed')
    if sha256_file(old/'quality_exclusions.json')!=plan['quality_policy_sha256']:raise ValueError('Quality policy changed')
    validate_source(cfg)
    originals={r['sample_id']:r for r in [json.loads(s) for s in (old/'manifest.jsonl').read_text().splitlines()]}
    if len(originals)!=2744:raise ValueError('Original sample count changed')
    cache={}
    for cache_root in (root/'runs/eight_frame_quarter_pilot_32',root/'runs/four_layout_independent_128'):
        snapshot=json.loads((cache_root/'dataset.json').read_text())['config']
        expected=json_safe(asdict(cfg))
        if any(snapshot[k]!=expected[k] for k in ('zemax','model','simulation')):raise ValueError('Pilot cache optical config differs')
        mapping=[snapshot['dataset']['positions_mm'].index(list(p)) for p in cfg.dataset.positions_mm]
        for line in (cache_root/'manifest.jsonl').read_text().splitlines():
            record=json.loads(line);sid=record.get('original_sample_id',record['sample_id'].rsplit('_',1)[0])
            if sid in originals:cache[sid]=(cache_root/'samples'/record['sample_id'],record,mapping)
    writer=DatasetWriter(root/plan['proposed_new_dataset'],resume=True)
    acceptance=load_legacy_acceptance(root,args.accept_known_photometry,plan_path,cfg_path,writer.root)
    prepared=writer.root/'prepared_model.zmx'
    snapshot={'config':json_safe(asdict(cfg)),'release_plan_sha256':sha256_file(plan_path),
        'original_manifest_sha256':plan['original_manifest_sha256'],'copy_GT_A_and_source':True}
    session=None
    try:
        if writer.is_new:
            session=connect(cfg.zemax.install_dir);prepare_shift_model(session,cfg,prepared)
            snapshot['prepared_model_sha256']=sha256_file(prepared);atomic_json(writer.root/'dataset.json',snapshot)
            shutil.copy2(old/'quality_exclusions.json',writer.root/'quality_exclusions.json')
        else:
            snapshot['prepared_model_sha256']=sha256_file(prepared)
            if json.loads((writer.root/'dataset.json').read_text())!=snapshot:raise ValueError('Resume configuration changed')
        completed=writer.completed_ids()
        candidates=[sid for sid,r in originals.items() if (r['split']=='test')==(args.stage=='test')]
        if args.stage=='canary':
            train=[sid for sid in sorted(candidates) if originals[sid]['split']=='train' and sid not in plan['excluded_ids']]
            stress=[sid for sid in ('0637_03','0337_00','0439_00','0184_03') if sid in train]
            cached=list(dict.fromkeys(stress+[sid for sid in train if sid in cache]))[:8]
            candidates=cached+[sid for sid in train if sid not in cache][:8]
            atomic_json(writer.root/'canary_ids.json',candidates)
        else:candidates=sorted(candidates)
        for sid in candidates:
            original=originals[sid];source=old/'samples'/sid;final=writer.root/'samples'/sid
            if sid in completed:continue
            # Recover a committed directory if interruption happened between rename and manifest append.
            if final.exists():
                record=json.loads((final/'metadata.json').read_text())
                check=verify_sample(final,record,source,original,plan['lens_positions_mm'],plan['excluded_ids'])
                check=apply_brightness_acceptance(check,acceptance)
                writer.append(record);completed.add(sid)
                if check['brightness_blocking'] and not acceptance:raise RuntimeError('Brightness gate failed on recovered '+sid)
                if check['brightness_blocking']:print('warning: preserved known photometry response '+sid,flush=True)
                continue
            partial=writer.root/'samples'/('.'+sid+'.partial');partial.mkdir(parents=True,exist_ok=True)
            for group in ('a','b'):(partial/group).mkdir(exist_ok=True)
            shutil.copy2(source/'source.png',partial/'source.png');shutil.copy2(source/'target.png',partial/'target.png')
            for i in range(4):shutil.copy2(source/'a'/f'frame_{i:02d}.png',partial/'a'/f'frame_{i:02d}.png')
            progress_path=partial/'frame_progress.json'
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            reused=False;cache_info=cache.get(sid)
            if cache_info:
                location,metadata,mapping=cache_info
                if sha256_file(location/'source.png')!=sha256_file(source/'source.png') or metadata['target_sha256']!=original['target_sha256']:
                    raise ValueError('Pilot cache source or GT differs for '+sid)
                for i,j in enumerate(mapping):
                    file=location/'b'/f'frame_{j:02d}.png'
                    if sha256_file(file)!=metadata['output_sha256']['b'][j]:raise ValueError('Pilot cache hash mismatch')
                    shutil.copy2(file,partial/'b'/f'frame_{i:02d}.png')
                reused=True
            else:
                if session is None:
                    session=connect(cfg.zemax.install_dir);session.system.LoadFile(str(prepared),False)
                staged=cfg.zemax.images_dir/('sr_quarter_release_'+sid+'.png')
                if staged.exists() and sha256_file(staged)!=sha256_file(source/'source.png'):
                    raise ValueError('Conflicting staging image '+str(staged))
                shutil.copy2(source/'source.png',staged)
                try:
                    for i,xy in enumerate(cfg.dataset.positions_mm):
                        target=partial/'b'/f'frame_{i:02d}.png'
                        if str(i) in progress and target.exists() and sha256_file(target)==progress[str(i)]:continue
                        # Uncommitted simulator output belongs only to this private partial sample.
                        if target.exists():target.unlink()
                        set_lens_shift(session,cfg,*xy);run_image_simulation(session,staged.name,target,cfg.simulation)
                        _normalize_grayscale(target);_verify_size(target,128)
                        progress[str(i)]=sha256_file(target);atomic_json(progress_path,progress)
                finally:staged.unlink(missing_ok=True)
            record={**original,'positions_b_mm':plan['lens_positions_mm'],'source_png_sha256':sha256_file(partial/'source.png'),
                'output_sha256':{g:[sha256_file(partial/g/f'frame_{i:02d}.png') for i in range(4)] for g in ('a','b')},
                'quarter_release_provenance':{'cache_reused':reused,'original_sample_id':sid,'plan_sha256':sha256_file(plan_path)}}
            if acceptance:record['quarter_release_provenance']['legacy_continuation_policy_sha256']=acceptance
            check=verify_sample(partial,record,source,original,plan['lens_positions_mm'],plan['excluded_ids'])
            check=apply_brightness_acceptance(check,acceptance)
            atomic_json(partial/'metadata.json',record);atomic_json(partial/'brightness.json',check)
            partial.rename(final);writer.append(record);completed.add(sid)
            atomic_json(writer.root/'generation_progress.json',{'stage':args.stage,'completed_total':len(completed),
                'last_sample':sid,'cache_reused':reused,'brightness':check})
            print(f'completed {len(completed)}/2744: {sid}; cached={reused}; mean_range={check["b_mean_range"]:.5f}',flush=True)
            if check['brightness_blocking'] and not acceptance:raise RuntimeError('Brightness gate >0.05 for '+sid+'; inspect without excluding it')
            if check['brightness_blocking']:print('warning: preserved known photometry response '+sid+'; no correction or exclusion',flush=True)
        atomic_json(writer.root/(args.stage.replace('-','_')+'_generation_complete.json'),{'completed_total':len(completed),'stage':args.stage})
    finally:
        if session is not None:session.close()


if __name__=='__main__':main()
