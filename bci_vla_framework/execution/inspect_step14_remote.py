"""Read-only SSH/SFTP inventory. No service launch, inference or robot I/O."""
from pathlib import Path
import hashlib
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect

DEST=ROOT/'execution/14_remote_audit.json'
if DEST.exists():raise SystemExit('Existing audit preserved.')
result={'scope':'read_only_ssh_source_inventory_no_inference','robot_commands':0,'EEG_connections':0,
    'service_launches':0,'checkpoint_selected':None,'files':[]}
try:
    client=connect()
    try:
        ftp=client.open_sftp()
        paths=['/home/oyjt/Scene-aware-BCI-VLA/scripts/robot/RTC_inference.py',
            '/home/oyjt/Scene-aware-BCI-VLA/scripts/training/serve_policy.py',
            '/home/oyjt/Scene-aware-BCI-VLA/scripts/inference.py',
            '/home/oyjt/openpi/src/openpi/policies/aloha_policy.py',
            '/home/oyjt/openpi/src/openpi/policies/policy.py',
            '/home/oyjt/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py',
            '/home/oyjt/openpi/src/openpi/serving/websocket_policy_server.py']
        secret=re.compile(r'password|passwd|secret|token|api_key|sshpass|private_key',re.I)
        relevant=re.compile(r'action|state|images|image_keys|front|left|right|normaliz|rtc|metadata|resize|degree|radian|gripper|joint|stop|cancel|infer\(',re.I)
        for path in paths:
            entry={'path':path}
            try:
                with ftp.open(path,'rb') as source:data=source.read(300001)
                entry.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),truncated=len(data)>300000)
                entry['selected_lines']=[{'line':i,'text':line[:500]} for i,line in enumerate(data.decode('utf-8',errors='replace').splitlines(),1)
                    if relevant.search(line) and not secret.search(line)][:180]
            except OSError as exc:entry['error']=type(exc).__name__
            result['files'].append(entry)
        for key,command in [('repository_head','git -C /home/oyjt/Scene-aware-BCI-VLA rev-parse HEAD'),
            ('repository_status','git -C /home/oyjt/Scene-aware-BCI-VLA status --porcelain'),
            ('listening_tcp_ports','ss -lntH'),
            ('gpu_inventory','nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader')]:
            _,out,err=client.exec_command(command,timeout=15)
            result[key]={'stdout':out.read(40000).decode('utf-8',errors='replace'),'exit_code':out.channel.recv_exit_status()}
        ftp.close();result['status']='read_only_success'
    finally:client.close()
except Exception as exc:
    result.update(status='connection_or_inspection_failed',error_type=type(exc).__name__)
DEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result.get(k) for k in ['status','error_type','robot_commands','EEG_connections','service_launches','checkpoint_selected']},indent=2))
