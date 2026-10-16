"""Fingerprint the selected simulation baseline, independently of old paper assets."""
from pathlib import Path
import json,shlex,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
code='''import hashlib,json,subprocess
from pathlib import Path
root=Path('/home/oyjt/workspace/Qwen2.5-VL-GR00T-LIBERO-4in1')
files=[]
for p in [root/'checkpoints/steps_30000_pytorch_model.pt',root/'config.yaml',root/'dataset_statistics.json']:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
 files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
repos={}
for name,path in [('starVLA','/home/oyjt/starVLA'),('LIBERO','/home/oyjt/LIBERO')]:
 r=subprocess.run(['git','-C',path,'rev-parse','HEAD'],capture_output=True,text=True)
 repos[name]={'path':path,'commit':r.stdout.strip(),'exit_code':r.returncode}
print(json.dumps({'scope':'selected_existing_LIBERO_baseline_not_original_paper_checkpoint','files':files,'repositories':repos,'training_performed':False,'downloads':False}))
'''
target=ROOT/'execution/14_libero_assets.json';assert not target.exists()
c=connect()
try:
    _,out,err=c.exec_command('/usr/bin/python3 -c '+shlex.quote(code),timeout=60)
    data=out.read().decode();exitcode=out.channel.recv_exit_status()
    if exitcode!=0:raise RuntimeError('Asset fingerprint failed')
    result=json.loads(data);target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(result['files']),'repositories':result['repositories'],'scope':result['scope']}))
finally:c.close()
