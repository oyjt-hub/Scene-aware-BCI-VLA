"""Run one bounded isolated probe on existing server runtime; save raw evidence."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def main():
    io_only=sys.argv[1:]==['--io-check']
    trace=sys.argv[1:]==['--load-trace']
    precision=sys.argv[1:]==['--precision-check']
    mask=sys.argv[1:] in (['--mask-overlay'],['--precision-check'])
    tied=sys.argv[1:] in (['--tied-overlay'],['--mask-overlay'],['--precision-check'])
    status_only=sys.argv[1:]==['--status']
    fast=sys.argv[1:] in (['--fast-overlay'],['--load-trace'],['--tied-overlay'],['--mask-overlay'],['--precision-check'])
    overlay=fast or sys.argv[1:]==['--existing-overlay']
    if sys.argv[1:] not in ([],['--existing-overlay'],['--fast-overlay'],['--status'],['--io-check'],['--load-trace'],['--tied-overlay'],['--mask-overlay'],['--precision-check']):raise SystemExit('Invalid mode')
    output=HERE/('03_text_probe_precision_raw.json' if precision else '03_text_probe_mask_raw.json' if mask else '03_text_probe_tied_raw.json' if tied else '03_load_trace.json' if trace else '03_text_probe_fast_raw.json' if fast else '03_text_probe_overlay_raw.json' if overlay else '03_text_probe_raw.json')
    if output.exists() and not (status_only or io_only):raise SystemExit('Existing probe evidence must be preserved')
    spec=importlib.util.spec_from_file_location('private_remote_reader',ROOT/'.server-inspect/remote_readonly.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    client=mod.connect()
    try:
        if io_only:
            code="""from pathlib import Path
import json,time
p=Path('/home/oyjt/workspace/RLinf-RECAP/models/baseline_sft/RLinf-Pi05-LIBERO-SFT/model.safetensors')
rows=[]
with p.open('rb') as f:
 for offset in [1048576,p.stat().st_size//2,p.stat().st_size-16*1024**2]:
  start=time.monotonic();f.seek(offset);raw=f.read(16*1024**2);rows.append(dict(offset=offset,bytes=len(raw),seconds=time.monotonic()-start))
mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}
print(json.dumps(dict(scope='bounded read-only IO diagnostic, not model inference',reads=rows,memory=mem)))
"""
            _,stdout,stderr=client.exec_command("timeout --kill-after=2 20 python3 - <<'IOCHECK'\n"+code+"\nIOCHECK\n",timeout=25)
            raw=stdout.read().decode();err=stderr.read().decode();exit_code=stdout.channel.recv_exit_status()
            value=json.loads(raw) if raw.strip() else {'status':'timeout_or_error'}
            value.update(exit_code=exit_code,stderr=err)
            target=HERE/'03_io_diagnostic.json'
            if target.exists():raise SystemExit('Preserve IO diagnostic')
            target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(value));return
        if status_only:
            _,stdout,stderr=client.exec_command('ps -u oyjt -o pid,comm,etime,rss',timeout=15)
            print(stdout.read().decode()[-2500:]);print(stderr.read().decode());return
        payload=(HERE/'step03_text_probe.py').read_text(encoding='utf-8')
        overlay_env=' PYTHONPATH=/home/oyjt/Scene-aware-BCI-VLA-runs/step15C_full_20261016_v1/deps:/home/oyjt/openpi/src' if overlay else ''
        if fast:overlay_env+=' V4_SKIP_RANDOM_INIT=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_COMPILE_THREADS=1'
        if trace:overlay_env+=' V4_LOAD_TRACE=1'
        if tied:overlay_env+=' V4_RESTORE_DECLARED_TIE=1'
        if mask:overlay_env+=' V4_MATCH_MASK_DTYPE=1'
        if precision:overlay_env+=' V4_PRECISION_CHECK=1'
        duration='65' if trace else '420'
        command="cd /home/oyjt/openpi && timeout --kill-after=10 "+duration+" env CUDA_VISIBLE_DEVICES=4 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1"+overlay_env+" .venv/bin/python -u - <<'V4PROBE'\n"+payload+"\nV4PROBE\n"
        _,stdout,stderr=client.exec_command(command,timeout=450)
        raw=stdout.read().decode('utf-8',errors='replace');err=stderr.read().decode('utf-8',errors='replace');code=stdout.channel.recv_exit_status()
        marker='V4_STEP03_JSON='
        result=json.loads(raw.rsplit(marker,1)[1]) if marker in raw else {'execution_status':'failed','error':'No structured probe output'}
        result.update(remote_exit_code=code,existing_overlay_used=overlay,stdout_before_marker=raw.split(marker)[0][-4000:],stderr=err[-8000:])
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'saved':str(output),'status':result['execution_status'],'error':result.get('error'),'stages':[s['name'] for s in result.get('stages',[])]}))
    finally:client.close()
if __name__=='__main__':main()
