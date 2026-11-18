"""Reproduce an existing B burst with the unchanged legacy rendering function."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from super_resolution_imaging.config import load_config
from super_resolution_imaging.pipeline import _normalize_grayscale
from super_resolution_imaging.data.writer import sha256_file
from super_resolution_imaging.data.quarter_release import atomic_json
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift,validate_source
from super_resolution_imaging.zos.image_simulation import run_image_simulation

ROOT=Path(__file__).resolve().parents[1]
cfg_path=ROOT/'configs/zemax_quarter_four_frozen.toml';cfg=load_config(cfg_path);validate_source(cfg)
dataset=ROOT/'runs/div2k_quarter_four_v1';prepared=dataset/'prepared_model.zmx'
snapshot=json.loads((dataset/'dataset.json').read_text(encoding='utf-8'))
if sha256_file(prepared)!=snapshot['prepared_model_sha256']:raise RuntimeError('Prepared model changed')
sample=dataset/'samples/0276_03'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
folder=ROOT/'runs/legacy_quarter_reproduction'/stamp;folder.mkdir(parents=True)
report={'completed':False,'passed':False,'sample':'0276_03','config_sha256':sha256_file(cfg_path),
        'prepared_model_sha256':sha256_file(prepared),'frames':[],'original_images_modified':False}
def save():atomic_json(folder/'report.json',report)
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(prepared),False)
    report['huygens_integral_method']=str(session.system.SystemData.Advanced.HuygensIntegralMethod)
    probe=session.system.Analyses.New_Analysis_SettingsFirst(session.ZOSAPI.Analysis.AnalysisIDM.ImageSimulation)
    try:
        s=probe.GetSettings()
        report['legacy_defaults']={'wavelength_rgb':bool(s.GetWavelengthRGB()),'relative_illumination':bool(s.UseRelativeIllumination),
                                   'fixed_apertures':bool(s.ApplyFixedApertures),'guard_band':str(s.GuardBand),'oversampling':str(s.Oversampling)}
    finally:probe.Close()
    staged=cfg.zemax.images_dir/('sr_legacy_reproduction_'+stamp+'.png')
    if staged.exists():raise FileExistsError(staged)
    shutil.copy2(sample/'source.png',staged)
    try:
        for i,position in enumerate(cfg.dataset.positions_mm):
            set_lens_shift(session,cfg,*position);target=folder/f'frame_{i:02d}.png'
            run_image_simulation(session,staged.name,target,cfg.simulation)
            _normalize_grayscale(target)
            row={'frame':i,'sha256':sha256_file(target),'original_sha256':sha256_file(sample/'b'/f'frame_{i:02d}.png')}
            row['identical']=row['sha256']==row['original_sha256'];report['frames'].append(row);save();print(row,flush=True)
    finally:staged.unlink(missing_ok=True)
report['completed']=True;report['passed']=all(r['identical'] for r in report['frames']);save()
atomic_json(ROOT/'runs/legacy_quarter_reproduction/latest.json',{'report':str(folder/'report.json'),**report})
if not report['passed']:raise RuntimeError('Legacy reproduction differs; do not mix outputs')
print('Legacy four-frame reproduction passed: 4/4 exact hashes',flush=True)
