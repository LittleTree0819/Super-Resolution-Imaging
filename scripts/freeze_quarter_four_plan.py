"""Write an auditable release specification only; never generates or trains."""
import json
from pathlib import Path
from super_resolution_imaging.config import load_config
from super_resolution_imaging.training.cli import sha256_file
from super_resolution_imaging.training.dataset import DEFAULT_EXCLUDED_IDS

root=Path(__file__).resolve().parents[1]
config_path=root/'configs/zemax_quarter_four_frozen.toml';config=load_config(config_path)
dataset=root/'runs/div2k_train_v1'
records=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
counts={split:sum(r['split']==split for r in records) for split in ('train','validation','test')}
effective={split:sum(r['split']==split and r['sample_id'] not in DEFAULT_EXCLUDED_IDS for r in records) for split in counts}
if counts!={'train':2592,'validation':72,'test':80}:raise ValueError('Dataset split changed')
if effective!={'train':2586,'validation':72,'test':80}:raise ValueError('Quality policy changed')
plan={'status':'frozen_design_only_not_generated','frozen_date':'2026-10-18',
    'decision':'Proceed toward a separately versioned new-layout dataset after staged QC; no generation in this action',
    'new_config':str(config_path),'new_config_sha256':sha256_file(config_path),
    'original_dataset':str(dataset),'original_manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
    'quality_policy_sha256':sha256_file(dataset/'quality_exclusions.json'),
    'raw_sample_count':2744,'raw_split_counts':counts,'effective_split_counts':effective,
    'excluded_ids':list(DEFAULT_EXCLUDED_IDS),'proposed_new_dataset':'runs/div2k_quarter_four_v1',
    'crop_gt_policy':'Use exact existing sample IDs, original source PNG bytes, GT bytes, source grouping and split; never re-crop, re-hash split, rotate or brightness-normalize per image',
    'a_group':'Reuse existing center frame repeated four times; A data unchanged',
    'b_group':'Exactly four frozen quarter positions in specified order; no center frame appended',
    'lens_positions_mm':[list(p) for p in config.dataset.positions_mm],
    'target_image_shifts_lr_px':[[.25,.25],[-.25,.25],[-.25,-.25],[.25,-.25]],
    'base_reference':'B first quarter-position frame bicubic; centered GT unchanged; no oracle alignment/brightness correction',
    'training':{'architecture':'BurstUNet2x base24 split residual sigma4 input4','batch_size':8,'patch_size':None,
        'optimizer':'AdamW','weight_decay':1e-5,'note_weight_decay':'Retained for comparability; negligible float32 decay at this lr',
        'loss':'L1+0.2*(1-SSIM)+0.2*component','global_brightness_weight':0.,'local_low_frequency_weight':0.,
        'A_initialization':{'path':'runs/train_a_expanded_split_residual_ssim_w020_stage1/best.pt',
            'sha256':'86afbf8a198b8c9565c0aff39b856751301d30242baf8f13611e262f9684a86e',
            'reason':'Exact same center A images and original source/split; do not reuse pilot B weights'},
        'B_initial_lr':1e-4,'B_epochs':20,'B_updates_per_epoch':324,'B_total_updates':6480,
        'scheduler':{'kind':'ReduceLROnPlateau','monitor':'full_validation_l1','patience_epochs':5,'factor':.5,'min_lr':1e-6},
        'early_stopping':False,'primary_seed':5400,'optional_replication_seeds':[5401,5402],
        'primary_comparison':'Matched final epoch20 vs original-layout same-budget control',
        'deployment_selection':'Highest full-validation PSNR, evaluate every epoch, include epoch0, exact ties retain earlier epoch',
        'metric_reporting':'Full image PSNR/SSIM/L1 primary; crop16 brightness/local low/high errors supplementary',
        'note':'Formal rule retains prior full-image selection, not pilot crop16 best criterion'},
    'staged_qc':['Reuse existing validated quarter frames for exact matching source crops where hashes agree; simulate remaining crops',
        'First check a train-only subset and known photometric stress cases; retain low-information and degraded sources',
        'Verify all sizes, source/target/frame hashes and manifest counts; new-frame within-burst max mean range >0.05 blocks pending diagnosis, not automatic exclusion',
        'Log center-referenced brightness diagnostics where center is available; do not treat content shifts or PSF changes as pure exposure drift',
        'Generate training and formal validation with frozen membership; handle formal test generation in a separate sealed stage',
        'Do not compute formal test image metrics or inspect test cases before checkpoint and selection rule locked'],
    'new_model_acceptance':'Report paired full-validation PSNR/SSIM and brightness/high-frequency tradeoffs vs original; no guarantee of superiority from pilot',
    'no_generation_or_training_started':True}
path=root/'configs/quarter_four_release_plan.json'
if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Existing frozen plan differs; use a new version')
path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'plan':str(path),'raw_split_counts':counts,'effective_split_counts':effective},indent=2))
