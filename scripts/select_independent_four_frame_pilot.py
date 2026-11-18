"""Freeze 128 new source-separated training-pool scenes for layout replication."""
import json
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from super_resolution_imaging.training.cli import sha256_file

root=Path(__file__).resolve().parents[1]
dataset=root/'runs/div2k_train_v1';out=root/'data/four_layout_independent_128'
if out.exists():raise FileExistsError(out)
old=json.loads((root/'data/nine_frame_pilot_sources/selection.json').read_text())['samples']
old_sources={r['original_sample_id'].split('_')[0] for r in old}
excluded=old_sources|{'0036','0449','0552'}
records=[json.loads(line) for line in (dataset/'manifest.jsonl').read_text().splitlines()]
by_source={}
protected={r['sample_id'].split('_')[0] for r in records if r['split']!='train'}
for r in records:
    sid=r['sample_id'];origin=sid.split('_')[0]
    if r['split']=='train' and origin not in excluded and origin not in protected:
        by_source.setdefault(origin,[]).append(r)
rng=random.Random(5417);candidates=[]
for origin in sorted(by_source):
    r=rng.choice(sorted(by_source[origin],key=lambda r:r['sample_id']))
    with Image.open(dataset/'samples'/r['sample_id']/'target.png') as im:
        std=float(np.asarray(im,dtype=np.float32).std()/255.)
    candidates.append((std,origin,r))
candidates.sort(key=lambda x:(x[0],x[1]))
if len(candidates)<128:raise ValueError('Not enough independent sources')
# Sample one source from each stratum; retain low-information scenes rather than post-hoc filtering.
edges=np.linspace(0,len(candidates),129,dtype=int)
chosen=[candidates[rng.randrange(int(edges[i]),int(edges[i+1]))] for i in range(128)]
out.mkdir(parents=True)
selection=[]
for rank,(std,origin,r) in enumerate(chosen):
    sid=r['sample_id'];folder=dataset/'samples'/sid
    source=out/f'{sid}.png';shutil.copy2(folder/'source.png',source)
    selection.append({'rank':rank,'sample_id':sid+'_00','original_sample_id':sid,'source_id':origin,
        'pilot_source':source.name,'split':'validation' if rank%4==2 else 'train','target_std':std,
        'source_png_sha256':sha256_file(source),'target_sha256':r['target_sha256'],
        'original_four_sha256':r['output_sha256']['b'],'original_record':r})
plan={'count':128,'seed':5417,'train_count':96,'validation_count':32,'samples':selection,
      'excluded_previous_pilot_sources':sorted(old_sources),'formal_validation_test_sources_untouched':True,
      'original_manifest_sha256':sha256_file(dataset/'manifest.jsonl')}
(out/'selection.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'selected':128,'training':96,'validation':32,'previous_source_overlap':0,
                  'candidate_sources':len(candidates)},indent=2))
