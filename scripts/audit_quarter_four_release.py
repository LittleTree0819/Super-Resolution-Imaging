"""Image-only brightness/hash/dimension QC; never loads a network checkpoint."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from super_resolution_imaging.data.writer import sha256_file
from super_resolution_imaging.data.quarter_release import verify_sample,atomic_json,load_legacy_acceptance,apply_brightness_acceptance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('existing','canary','train-validation','all'),required=True)
    parser.add_argument('--accept-known-photometry',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'configs/quarter_four_release_plan.json').read_text());old=Path(plan['original_dataset'])
    new=root/plan['proposed_new_dataset'];originals={r['sample_id']:r for r in [json.loads(s) for s in (old/'manifest.jsonl').read_text().splitlines()]}
    acceptance=load_legacy_acceptance(root,args.accept_known_photometry,root/'configs/quarter_four_release_plan.json',Path(plan['new_config']),new)
    records=[json.loads(s) for s in (new/'manifest.jsonl').read_text().splitlines()]
    if len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate IDs')
    if sha256_file(old/'manifest.jsonl')!=plan['original_manifest_sha256']:raise ValueError('Original manifest changed')
    if sha256_file(new/'quality_exclusions.json')!=plan['quality_policy_sha256']:raise ValueError('Quality policy changed')
    if args.stage=='existing':expected={r['sample_id'] for r in records}
    elif args.stage=='canary':expected=set(json.loads((new/'canary_ids.json').read_text()))
    else:expected={sid for sid,r in originals.items() if args.stage=='all' or r['split']!='test'}
    if not expected<={r['sample_id'] for r in records}:raise ValueError('Missing expected samples')
    if args.stage=='all' and len(records)!=2744:raise ValueError('Final count must be 2744')
    if args.stage=='all' and list((new/'samples').glob('.*.partial')):raise ValueError('Unfinished partial samples remain')
    checked=[]
    for r in records:
        if r['sample_id'] not in expected:continue
        sid=r['sample_id']
        checked.append(verify_sample(new/'samples'/sid,r,old/'samples'/sid,originals[sid],plan['lens_positions_mm'],plan['excluded_ids']))
    checked=[apply_brightness_acceptance(row,acceptance) for row in checked]
    bad=[r for r in checked if r['brightness_blocking']]
    counts=dict(Counter(r['split'] for r in checked))
    effective=dict(Counter(r['split'] for r in checked if not r['excluded_by_frozen_policy']))
    if args.stage=='all' and (counts!=plan['raw_split_counts'] or effective!=plan['effective_split_counts']):
        raise ValueError('Raw or effective split counts changed')
    summary={'stage':args.stage,'samples_checked':len(checked),'decoded_png_count':10*len(checked),
        'split_counts':counts,'effective_split_counts':effective,
        'brightness_blocking_count':len(bad),'blocking_ids':[r['sample_id'] for r in bad],
        'max_b_mean_range':max(r['b_mean_range'] for r in checked),
        'max_center_mean_difference':max(r['max_b_center_mean_difference'] for r in checked),
        'sha256_and_dimensions':'passed','GT_source_A_preservation':'passed','training_started':False,
        'test_handling':'Image integrity/brightness QC only; no model metrics or test image previews',
        'passed':not bad,'integrity_passed':True,'brightness_gate_passed':not bad,
        'release_allowed':not bad or bool(acceptance),'legacy_continuation_policy_sha256':acceptance,
        'optical_numerical_validation':'not certified by this image-only audit'}
    with (new/f'brightness_{args.stage}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(checked[0]));w.writeheader();w.writerows(checked)
    atomic_json(new/f'qc_{args.stage}.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    atomic_json(new/f'photometry_flags_{args.stage}.json',bad)
    if bad and not acceptance:raise RuntimeError('Brightness QC blocks release; preserve scenes for diagnosis')
    if bad:print(f'warning: {len(bad)} photometry responses retained under explicit continuation policy',flush=True)
    if args.stage=='all':
        atomic_json(new/'COMPLETE.json',{'samples':2744,
            'qc':'integrity_passed_with_known_optical_limitations' if acceptance else 'passed',
            'brightness_gate_passed':not bad,'accepted_photometry_warning_count':len(bad) if acceptance else 0,
            'legacy_continuation_policy_sha256':acceptance,'manifest_sha256':sha256_file(new/'manifest.jsonl'),
            'frozen_split_preserved':True,'no_training_started':True})


if __name__=='__main__':main()
