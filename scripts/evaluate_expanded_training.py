"""Evaluate locked expanded checkpoints against the original model, without training."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model, confidence_interval, tensor_image
from super_resolution_imaging.training.baselines import bicubic, four_frame_mean, metrics


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/expanded_training_evaluation_20261011'
    output.mkdir(exist_ok=False)
    paths = {
        'old_b': root / 'runs/train_b_split_residual_ssim_w020_stage2/best.pt',
        'new_a': root / 'runs/train_a_expanded_split_residual_ssim_w020_stage1/best.pt',
        'new_b': root / 'runs/train_b_expanded_split_residual_ssim_w020_stage2/best.pt',
    }
    models = {name: load_model(path, 'a' if name == 'new_a' else 'b', torch.device('cpu'))[0]
              for name, path in paths.items()}
    report = {'checkpoints': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                              for k, p in paths.items()}, 'splits': {}}
    for split in ('validation', 'test'):
        datasets = {g: ZemaxBurstDataset(root / 'runs/div2k_train_v1', g, split,
                    exclude_ids=DEFAULT_EXCLUDED_IDS) for g in ('a', 'b')}
        assert [r['sample_id'] for r in datasets['a'].records] == [r['sample_id'] for r in datasets['b'].records]
        loaders = {g: DataLoader(ds, batch_size=8, shuffle=False) for g, ds in datasets.items()}
        rows, examples = [], []
        with torch.inference_mode():
            for a, b in zip(loaders['a'], loaders['b']):
                assert a['sample_id'] == b['sample_id']
                assert torch.equal(a['target'], b['target'])
                preds = {'bicubic': bicubic(b['input'][:, :1]), 'mean': four_frame_mean(b['input']),
                         'old_b': models['old_b'](b['input']), 'new_a': models['new_a'](a['input']),
                         'new_b': models['new_b'](b['input'])}
                scores = {k: metrics(v, b['target']) for k, v in preds.items()}
                for i, sid in enumerate(b['sample_id']):
                    row = {'sample_id': sid}
                    for name, vals in scores.items():
                        for metric, val in zip(('l1', 'psnr_db', 'ssim'), vals):
                            row[f'{name}_{metric}'] = float(val[i])
                    rows.append(row)
                    examples.append({'sample_id': sid, 'target': b['target'][i, 0].clone(),
                                     **{k: v[i, 0].clone() for k, v in preds.items()}})
        with (output / f'{split}_per_sample.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = {'samples': len(rows), 'metrics': {}, 'paired': {}}
        for name in preds:
            result['metrics'][name] = {m: float(np.mean([r[f'{name}_{m}'] for r in rows]))
                                      for m in ('l1', 'psnr_db', 'ssim')}
        for reference in ('old_b', 'new_a', 'bicubic'):
            result['paired'][f'new_b_minus_{reference}'] = {}
            for m in ('l1', 'psnr_db', 'ssim'):
                delta = np.array([r[f'new_b_{m}'] - r[f'{reference}_{m}'] for r in rows])
                result['paired'][f'new_b_minus_{reference}'][m] = {
                    'mean': float(delta.mean()), 'ci95_mean': confidence_interval(delta),
                    'better': int(((delta < 0) if m == 'l1' else (delta > 0)).sum()),
                    'median': float(np.median(delta))}
        ordered = sorted(range(len(rows)), key=lambda i: rows[i]['new_b_psnr_db'] - rows[i]['old_b_psnr_db'])
        result['worst_changes'] = [rows[i] for i in ordered[:3]]
        result['best_changes'] = [rows[i] for i in ordered[-3:][::-1]]
        chosen = ordered[-2:][::-1] + ordered[:2]
        labels = ('bicubic', 'old_b', 'new_a', 'new_b', 'target')
        canvas = Image.new('L', (256 * len(labels), 280 * len(chosen)), 255)
        draw = ImageDraw.Draw(canvas)
        for row, index in enumerate(chosen):
            example = examples[index]
            for col, name in enumerate(labels):
                canvas.paste(tensor_image(example[name]), (col * 256, row * 280 + 24))
                draw.text((col * 256 + 4, row * 280 + 4), f"{example['sample_id']} {name}", fill=0)
        canvas.save(output / f'{split}_best_worst.png')
        report['splits'][split] = result
        print(split, json.dumps(result['metrics']), flush=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
