"""Hash the second backend and checked simulator conversion sources."""
from pathlib import Path
import json,shlex,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
code='''import hashlib,json,ast
from pathlib import Path
root=Path('/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT')
paths=[root/'model.safetensors',root/'assets/physical-intelligence/libero/norm_stats.json',
 Path('/home/oyjt/openpi/src/openpi/training/config.py'),Path('/home/oyjt/openpi/src/openpi/policies/libero_policy.py'),
 Path('/home/oyjt/workspace/miniconda3/envs/libero/lib/python3.10/site-packages/robosuite/controllers/base_controller.py')]
files=[]
for p in paths:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
 files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
text=paths[-1].read_text();tree=ast.parse(text)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='scale_action')
print(json.dumps({'scope':'pi05_existing_simulation_baseline_not_original_paper_checkpoint','files':files,
 'controller_conversion_source':ast.get_source_segment(text,node),'downloads_or_training_performed':False}))
'''
target=ROOT/'execution/15_pi05_assets.json';assert not target.exists()
c=connect()
try:
    _,out,err=c.exec_command('/usr/bin/python3 -c '+shlex.quote(code),timeout=60)
    data=out.read().decode();status=out.channel.recv_exit_status()
    if status:raise RuntimeError('Asset fingerprint failed')
    value=json.loads(data);target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print('Second-backend fingerprints and controller reference saved.')
finally:c.close()
