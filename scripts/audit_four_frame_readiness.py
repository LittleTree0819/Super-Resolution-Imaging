"""Read-only full-file audit of the expanded four-frame training dataset."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from PIL import Image
from super_resolution_imaging.training.dataset import DEFAULT_EXCLUDED_IDS
from verify_training_expansion import frozen_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('runs/div2k_train_v1'))
    p.add_argument('--output',type=Path,default=Path('runs/four_frame_readiness_20261009.json'))
    args=p.parse_args(); root=args.dataset
    rows=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
    errors=[]; counts=Counter(r['split'] for r in rows)
    if counts!={'train':2592,'validation':72,'test':80}: errors.append(f'Unexpected counts: {counts}')
    if len({r['sample_id'] for r in rows})!=len(rows): errors.append('Duplicate sample IDs')
    complete=json.loads((root/'COMPLETE.json').read_text())
    if complete['samples']!=len(rows): errors.append('Completion count mismatch')
    policy=json.loads((root/'quality_exclusions.json').read_text())
    if tuple(policy['excluded_sample_ids'])!=DEFAULT_EXCLUDED_IDS: errors.append('Exclusion policy mismatch')
    if not set(DEFAULT_EXCLUDED_IDS)<={r['sample_id'] for r in rows}: errors.append('Excluded sample missing from manifest')
    effective=Counter(r['split'] for r in rows if r['sample_id'] not in DEFAULT_EXCLUDED_IDS)
    sources=defaultdict(set); crops=defaultdict(set); duplicates=[]; checked=0
    for i,r in enumerate(rows):
        sid=r['sample_id']; folder=root/'samples'/sid
        try:
            if json.loads((folder/'metadata.json').read_text())!=r: errors.append(f'Metadata mismatch {sid}')
            if r['input_shape']!=[4,128,128] or r['target_shape']!=[1,256,256]: errors.append(f'Bad tensor shape {sid}')
            if len(set(r['output_sha256']['a']))!=1: errors.append(f'A not identical {sid}')
            if r['output_sha256']['a'][0]!=r['output_sha256']['b'][0]: errors.append(f'A/B center mismatch {sid}')
            for group in ('a','b'):
                if len(r['output_sha256'][group])!=4: errors.append(f'Wrong frame count {sid}')
                for f in range(4):
                    path=folder/group/f'frame_{f:02d}.png'
                    if sha256(path.read_bytes()).hexdigest()!=r['output_sha256'][group][f]: errors.append(f'Frame hash mismatch {path}')
                    with Image.open(path) as im:
                        im.load()
                        if im.mode!='L' or im.size!=(128,128): errors.append(f'Invalid frame {path}')
                    checked+=1
            target=folder/'target.png'
            if sha256(target.read_bytes()).hexdigest()!=r['target_sha256']: errors.append(f'Target hash mismatch {sid}')
            for path,size in ((target,256),(folder/'source.png',768)):
                with Image.open(path) as im:
                    im.load()
                    if im.mode!='L' or im.size!=(size,size): errors.append(f'Invalid image {path}')
                checked+=1
        except Exception as exc: errors.append(f'{sid}: {type(exc).__name__}: {exc}')
        sources[r['source_sha256']].add(r['split'])
        box=tuple(r['crop_box'])
        if box in crops[r['source_sha256']]: duplicates.append(sid)
        crops[r['source_sha256']].add(box)
        if (i+1)%500==0: print(f'audited {i+1}/{len(rows)}',flush=True)
    leaks=[s for s,splits in sources.items() if len(splits)>1]
    if leaks: errors.append(f'Source leakage across splits: {len(leaks)}')
    baseline=json.loads((root/'expansion_baseline.json').read_text())
    frozen_ok=frozen_digest(rows)==baseline['frozen_digest']
    if not frozen_ok: errors.append('Frozen validation/test records changed')
    declared={r['sample_id'] for r in rows}
    directories=list((root/'samples').iterdir())
    unexpected=[p.name for p in directories if p.is_dir() and p.name not in declared]
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if not errors else 'failed','training_started':False,'raw_counts':dict(counts),'effective_counts':dict(effective),'excluded_ids':list(DEFAULT_EXCLUDED_IDS),'files_decoded':checked,'frame_hashes_checked':len(rows)*8,'target_hashes_checked':len(rows),'unique_original_sources':len(sources),'source_split_leaks':leaks,'duplicate_crop_ids':duplicates,'frozen_evaluation_unchanged':frozen_ok,'unexpected_sample_directories':unexpected,'free_disk_gb':shutil.disk_usage(root).free/1024**3,'manifest_sha256':sha256((root/'manifest.jsonl').read_bytes()).hexdigest(),'errors':errors}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if errors: raise SystemExit(2)


if __name__=='__main__': main()
