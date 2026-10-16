"""Collect all owned full-pipeline jobs, preserving failed prototypes and calibration."""
from pathlib import Path
import hashlib,json,stat,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
folder=ROOT/'runs/pilot_robot_14C_full_pipeline';assert not folder.exists()
c=connect();files=[]
try:
    ftp=c.open_sftp()
    ftp.stat('/home/oyjt/Scene-aware-BCI-VLA-runs/step14C_full_20261016_v3/orchestration.json')
    folder.mkdir(parents=True)
    for version in ['v1','v2','v3']:
        job='/home/oyjt/Scene-aware-BCI-VLA-runs/step14C_full_20261016_'+version
        names=['orchestration.json','worker.log','server.log','launcher.log','14C_run_plan.json','sources.zip']
        try:names+=['results/'+e.filename for e in ftp.listdir_attr(job+'/results') if stat.S_ISREG(e.st_mode)]
        except FileNotFoundError:pass
        for name in names:
            try:
                meta=ftp.stat(job+'/'+name)
                if meta.st_size>200_000_000:raise RuntimeError('Unexpected artifact size')
                destination=folder/version/name;destination.parent.mkdir(parents=True,exist_ok=True);ftp.get(job+'/'+name,str(destination))
                files.append({'path':str(destination.resolve()),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'remote_path':job+'/'+name})
            except FileNotFoundError:pass
    ftp.close()
finally:c.close()
(folder/'collected_manifest.json').write_text(json.dumps({'scope':'actual_model_LIBERO_complete_functional_pipeline','files':files},indent=2)+'\n',encoding='utf-8')
summary=json.loads((folder/'v3/results/summary.json').read_text())
print(json.dumps({'all_checks_passed':summary['all_checks_passed'],'cases':[
    {k:e[k] for k in ['id','final_state','reason','checks','actual_sim_steps','wrong_target_start','sessions']} for e in summary['episodes']]},indent=2))
