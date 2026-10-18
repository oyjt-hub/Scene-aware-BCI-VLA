"""Read-only remote source/header audit; never executes model or device code."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PAYLOAD=r'''
from pathlib import Path
import hashlib,json,struct,subprocess
home=Path('/home/oyjt')
def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=12)
        return dict(code=r.returncode,stdout=r.stdout[:18000],stderr=r.stderr[:1000])
    except subprocess.TimeoutExpired:
        return dict(code=None,error='timeout',args=args)
roots=[home/'openpi',home/'workspace/openpi']
relative=['src/openpi/models/pi0.py','src/openpi/models/pi0_config.py','src/openpi/models/gemma.py','src/openpi/models/tokenizer.py','src/openpi/models_pytorch/pi0_pytorch.py','src/openpi/models_pytorch/gemma_pytorch.py','src/openpi/policies/policy_config.py','src/openpi/training/config.py','pyproject.toml']
files=[]
def read(p):
    if p.is_file() and p.stat().st_size<250000:
        raw=p.read_bytes();files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8',errors='replace')))
for p in [Path('/AGENTS.md'),Path('/home/AGENTS.md'),home/'AGENTS.md',home/'workspace/AGENTS.md']:
    read(p)
for root in roots:
    read(root/'AGENTS.md')
    for p in relative:read(root/p)
checkpoint=home/'workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT'
read(checkpoint/'config.json')
for name in ['_METADATA','_CHECKPOINT_METADATA']:
    read(home/'workspace/checkpoints/pi05_base/params'/name)
p=checkpoint/'model.safetensors'
with p.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    assert n<10000000
    header=json.loads(f.read(n))
environments=[]
for env in [home/'openpi/.venv',home/'workspace/openpi/.venv',home/'workspace/miniconda3/envs/libero']:
    for sp in env.glob('lib/python*/site-packages'):
        environments.append(dict(root=str(env),site_packages=str(sp),python_exists=(env/'bin/python').exists()))
        read(sp/'transformers/models/gemma/modeling_gemma.py')
        read(sp/'transformers/models/paligemma/modeling_paligemma.py')
result=dict(scope='read-only source and tensor metadata; no weight loading',sources=files,safetensors=dict(path=str(p),bytes=p.stat().st_size,header=header),environments=environments,gpu=command(['nvidia-smi','--query-gpu=index,name,memory.free,memory.used','--format=csv,noheader,nounits']),git={str(r):dict(head=command(['git','-C',str(r),'rev-parse','HEAD']),status=command(['git','-C',str(r),'status','--porcelain'])) for r in roots})
print(json.dumps(result))
'''

def main():
    output=OUT/'03_remote_source_audit.json'
    if output.exists():raise SystemExit('Preserve existing audit evidence')
    spec=importlib.util.spec_from_file_location('private_remote_reader',ROOT/'.server-inspect/remote_readonly.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=module.connect()
    try:
        _,stdout,stderr=client.exec_command("python3 - <<'PY'\n"+PAYLOAD+"\nPY\n",timeout=55)
        raw=stdout.read().decode();error=stderr.read().decode();code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError('Remote audit failed: '+error[:1500])
        result=json.loads(raw)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        folder=OUT/'03_sources';folder.mkdir(exist_ok=True)
        for i,row in enumerate(result['sources']):
            (folder/(str(i).zfill(2)+'_'+Path(row['path']).name)).write_text(row['text'],encoding='utf-8')
        print(json.dumps(dict(saved=str(output),source_count=len(result['sources']),environments=result['environments'],gpu=result['gpu'])))
    finally:client.close()

if __name__=='__main__':main()
