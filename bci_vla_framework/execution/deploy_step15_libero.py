"""Launch only the owned second-VLA simulator test in a new directory."""
from pathlib import Path
import json,sys,shlex
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
job='/home/oyjt/Scene-aware-BCI-VLA-runs/step15_libero_20261016_v4'
manifest=ROOT/'execution/15_remote_job_v4.json';assert not manifest.exists()
c=connect()
try:
    ftp=c.open_sftp()
    for directory in [job,job+'/bci_vla',job+'/bci_vla/pilot',job+'/bci_vla/adapters',job+'/bci_vla/adapters/vla']:
        ftp.mkdir(directory)
    for rel in ['bci_vla/__init__.py','bci_vla/protocols.py','bci_vla/pilot/__init__.py',
        'bci_vla/pilot/libero_gate.py','bci_vla/pilot/libero_actions.py',
        'bci_vla/adapters/__init__.py','bci_vla/adapters/vla/__init__.py','bci_vla/adapters/vla/fake.py','bci_vla/adapters/vla/libero_pi05.py']:
        ftp.put(str(ROOT/rel),job+'/'+rel)
    for rel in ['step15_pi05_orchestrator.py','step15_libero_worker.py']:
        ftp.put(str(ROOT/'execution'/rel),job+'/'+rel)
    ftp.close()
    command='cd '+shlex.quote(job)+' && nohup /usr/bin/timeout 1000s /usr/bin/python3 -u step15_pi05_orchestrator.py > launcher.log 2>&1 < /dev/null &'
    _,out,err=c.exec_command(command,timeout=10);status=out.channel.recv_exit_status()
    v={'scope':'owned_second_VLA_LIBERO_simulation','remote_job':job,'launch_exit':status,'physical_robot_commands':0,'real_BCI_trials':0}
    manifest.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
finally:c.close()
