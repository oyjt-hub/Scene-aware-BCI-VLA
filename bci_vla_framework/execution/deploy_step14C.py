"""Bundle an isolated full-framework simulator job; never edit old repositories."""
from pathlib import Path
import hashlib,json,sys,shlex,zipfile,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
job='/home/oyjt/Scene-aware-BCI-VLA-runs/step14C_full_20261016_v3';manifest=ROOT/'execution/14C_remote_job_v3.json';assert not manifest.exists()
orchestrator=(ROOT/'execution/step14_libero_orchestrator.py').read_text()
orchestrator=orchestrator.replace('16814','16816').replace('step14_libero_worker.py','step14C_worker.py').replace('timeout=660','timeout=840')
(ROOT/'execution/step14C_orchestrator.py').write_text(orchestrator,encoding='utf-8')
buffer=io.BytesIO();files=[]
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for directory,pattern in [('bci_vla','*.py'),('configs','*.json'),('examples','*.json')]:
        for p in (ROOT/directory).rglob(pattern):
            if '__pycache__' in p.parts:continue
            archive.write(p,p.relative_to(ROOT).as_posix());files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    for name in ['step14C_worker.py','step14C_engine.py','step14C_orchestrator.py','14C_run_plan_v3.json']:
        p=ROOT/'execution'/name;archive.write(p,'14C_run_plan.json' if name=='14C_run_plan_v3.json' else name);files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
c=connect()
try:
    ftp=c.open_sftp();ftp.mkdir(job)
    with ftp.open(job+'/sources.zip','wb') as f:f.write(buffer.getvalue())
    ftp.close()
    code="import zipfile;zipfile.ZipFile('sources.zip').extractall('.')"
    command='cd '+shlex.quote(job)+' && /usr/bin/python3 -c '+shlex.quote(code)+' && nohup /usr/bin/timeout 1100s /usr/bin/python3 -u step14C_orchestrator.py > launcher.log 2>&1 < /dev/null &'
    _,out,err=c.exec_command(command,timeout=10);status=out.channel.recv_exit_status()
    manifest.write_text(json.dumps({'remote_job':job,'launch_exit':status,'files':files,'archive_sha256':hashlib.sha256(buffer.getvalue()).hexdigest()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'remote_job':job,'launch_exit':status,'bundled_sources':len(files)}))
finally:c.close()
