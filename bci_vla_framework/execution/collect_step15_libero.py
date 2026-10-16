"""Collect owned second-VLA run artifacts after bounded worker completion."""
from pathlib import Path
import hashlib,json,stat,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
folder=ROOT/'runs/transfer_step15_pi05_final';assert not folder.exists()
c=connect();files=[]
try:
    ftp=c.open_sftp()
    ftp.stat('/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4/orchestration.json')
    folder.mkdir(parents=True)
    for version in ['v1','v2','v3','v4']:
        job='/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_'+version
        names=['orchestration.json','server.log','worker.log','launcher.log']
        try:
            names+=['results/'+e.filename for e in ftp.listdir_attr(job+'/results') if stat.S_ISREG(e.st_mode)]
        except FileNotFoundError:pass
        for name in names:
            try:
                meta=ftp.stat(job+'/'+name)
                if meta.st_size>100_000_000:raise RuntimeError('Unexpected artifact size')
                dest=folder/version/name;dest.parent.mkdir(parents=True,exist_ok=True);ftp.get(job+'/'+name,str(dest))
                files.append({'path':str(dest.resolve()),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'remote_path':job+'/'+name})
            except FileNotFoundError:pass
    executed=[]
    for rel,local in [('step15_libero_worker.py','execution/step15_libero_worker.py'),
        ('step15_pi05_orchestrator.py','execution/step15_pi05_orchestrator.py'),
        ('bci_vla/pilot/libero_gate.py','bci_vla/pilot/libero_gate.py'),
        ('bci_vla/adapters/vla/libero_pi05.py','bci_vla/adapters/vla/libero_pi05.py')]:
        remote='/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4/'+rel
        data=ftp.open(remote,'rb').read();fingerprint=hashlib.sha256(data).hexdigest()
        executed.append({'remote_path':remote,'local_path':str(ROOT/local),'sha256':fingerprint,
            'matches_local':fingerprint==hashlib.sha256((ROOT/local).read_bytes()).hexdigest()})
    ftp.close()
finally:c.close()
(folder/'collected_manifest.json').write_text(json.dumps({'scope':'owned_pi05_simulation_artifacts','files':files,'executed_sources':executed},indent=2)+'\n',encoding='utf-8')
s=json.loads((folder/'v4/results/summary.json').read_text())
print(json.dumps({'episodes':s['episodes'],'nominal_task_successes':s['nominal_task_successes'],
    'all_episodes_without_exception':s['all_episodes_without_exception'],'all_software_guards_passed':s['all_software_guards_passed']},indent=2))
