"""Upload an isolated simulation job; never edits the existing server repository."""
from pathlib import Path
import json,sys,posixpath,shlex
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
job='/home/oyjt/Scene-aware-BCI-VLA-runs/step14_libero_20261016_v3'
manifest=ROOT/'execution/14_remote_job_v3.json'
assert not manifest.exists()
c=connect()
try:
    ftp=c.open_sftp()
    for d in ['/home/oyjt/Scene-aware-BCI-VLA-runs',job,job+'/bci_vla',job+'/bci_vla/pilot',job+'/bci_vla/protocols']:
        try:ftp.mkdir(d)
        except OSError:
            if d==job:raise RuntimeError('Use new isolated job directory.')
    sources=[ROOT/'bci_vla/__init__.py',ROOT/'bci_vla/protocols.py',ROOT/'bci_vla/pilot/__init__.py',ROOT/'bci_vla/pilot/libero_gate.py',ROOT/'bci_vla/pilot/libero_actions.py']
    for path in sources:ftp.put(str(path),job+'/'+path.relative_to(ROOT).as_posix())
    for name in ['step14_libero_worker.py','step14_libero_orchestrator.py']:ftp.put(str(ROOT/'execution'/name),job+'/'+name)
    ftp.close()
    command='cd '+shlex.quote(job)+' && nohup /usr/bin/timeout 1000s /usr/bin/python3 -u step14_libero_orchestrator.py > launcher.log 2>&1 < /dev/null &'
    _,out,err=c.exec_command(command,timeout=10)
    status=out.channel.recv_exit_status()
    value={'scope':'isolated_LIBERO_simulation_job','remote_job':job,'launch_exit':status,'physical_robot_commands':0,'real_BCI_trials':0}
    manifest.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))
finally:c.close()
