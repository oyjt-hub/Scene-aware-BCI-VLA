"""Collect only owned migration runs, including unsuccessful runs."""
from pathlib import Path
import hashlib,json,stat,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'.server-inspect'))
from remote_readonly import connect
folder=ROOT/'runs/transfer_step15C_full';folder.mkdir(exist_ok=True)
manifest=folder/'collected_manifest.json';files=json.loads(manifest.read_text())['files'] if manifest.exists() else []
c=connect()
try:
 ftp=c.open_sftp()
 for jobfile in sorted((ROOT/'execution').glob('15C*remote_job*.json')):
  job=json.loads(jobfile.read_text())['remote_job'];name=job.rsplit('/',1)[-1];destination=folder/name
  if destination.exists():continue
  try:ftp.stat(job+'/orchestration.json')
  except FileNotFoundError:continue
  destination.mkdir()
  names=['orchestration.json','worker.log','server.log','launcher.log','15C_run_plan.json','sources.zip']
  try:names+=['results/'+x.filename for x in ftp.listdir_attr(job+'/results') if stat.S_ISREG(x.st_mode)]
  except FileNotFoundError:pass
  for name in names:
   try:
    size=ftp.stat(job+'/'+name).st_size;assert size<200_000_000
    p=destination/name;p.parent.mkdir(exist_ok=True,parents=True);ftp.get(job+'/'+name,str(p));files.append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   except FileNotFoundError:pass
 ftp.close()
finally:c.close()
manifest.write_text(json.dumps(dict(files=files),indent=2)+'\n')
for q in folder.glob('*/results/summary.json'):
 s=json.loads(q.read_text());print(q.parent.parent.name,s['all_checks_passed'],[(e['id'],e['final_state'],e['reason'],e['actual_sim_steps'],e.get('packet_events')) for e in s['episodes']])
