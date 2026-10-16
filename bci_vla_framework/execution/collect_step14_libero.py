"""Collect only the owned simulation job artifacts; no global process changes."""
from pathlib import Path
import hashlib,json,stat,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
folder=ROOT/'runs/pilot_robot_14_libero_simulation_v3';assert not folder.exists();folder.mkdir(parents=True)
c=connect();files=[]
try:
    ftp=c.open_sftp()
    for version in ['v3']:
        job='/home/oyjt/Scene-aware-BCI-VLA-runs/step14_libero_20261016_'+version
        for name in ['orchestration.json','server.log','worker.log','launcher.log']:
            try:
                meta=ftp.stat(job+'/'+name)
                if meta.st_size>20_000_000:raise RuntimeError('Unexpected log size')
                dest=folder/version/name;dest.parent.mkdir(parents=True,exist_ok=True);ftp.get(job+'/'+name,str(dest))
                files.append({'remote_path':job+'/'+name,'path':str(dest.resolve()),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
            except FileNotFoundError:pass
        if version=='v3':
            for entry in ftp.listdir_attr(job+'/results'):
                if not stat.S_ISREG(entry.st_mode) or entry.st_size>100_000_000:continue
                dest=folder/version/'results'/entry.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                ftp.get(job+'/results/'+entry.filename,str(dest))
                files.append({'remote_path':job+'/results/'+entry.filename,'path':str(dest.resolve()),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    ftp.close()
finally:c.close()
(folder/'collected_manifest.json').write_text(json.dumps({'scope':'owned_LIBERO_simulation_artifacts','files':files},indent=2)+'\n',encoding='utf-8')
summary=json.loads((folder/'v3/results/summary.json').read_text())
print(json.dumps({'episodes':[{k:e[k] for k in ['episode','task_id','language','task_success','termination','applied_actions','error','checks']} for e in summary['episodes']],
    'nominal_task_successes':summary['nominal_task_successes'],'all_software_guards_passed':summary['all_software_guards_passed'],
    'all_episodes_without_exception':summary['all_episodes_without_exception'],'elapsed_s':summary['elapsed_s']},indent=2))
