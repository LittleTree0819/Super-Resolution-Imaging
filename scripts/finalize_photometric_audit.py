"""One-shot postprocessor for the already-running diagnostic batch."""
import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--worker-pid',type=int,required=True);args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/photometric_forward_audit_v1'
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.restype=ctypes.c_void_p
kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x100000,False,args.worker_pid)
status={'worker_pid':args.worker_pid,'state':'waiting_for_image_sampling','training_started':False,'generation_resumed':False}
def save():
    status['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    p=OUT/'pipeline_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(status,indent=2),encoding='utf-8');t.replace(p)
previous=None
try:
    while True:
        report=json.loads((OUT/'image_sampling/report.json').read_text(encoding='utf-8'))
        rows=report['rows'];status['image_jobs_completed']=len(rows);status['image_jobs_planned']=12
        if rows:status['latest_result']={k:rows[-1][k] for k in ('group','sampling','mean')}
        if len(rows)!=previous:
            save();print(json.dumps(status),flush=True);previous=len(rows)
        if report['completed']:
            status['state']='finalizing';save()
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_photometric_audit.py')],cwd=ROOT,check=True)
            status['state']='completed_hold_production';save();print(json.dumps(status),flush=True);break
        if not handle or kernel.WaitForSingleObject(handle,0)!=258:
            status['state']='interrupted_before_completion';save()
            raise RuntimeError('Image audit worker exited before completed flag; outputs retained for resume')
        time.sleep(15)
finally:
    if handle:kernel.CloseHandle(handle)
