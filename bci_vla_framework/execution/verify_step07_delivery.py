"""Check offline scene/outcome delivery and reused-source identity only."""
from pathlib import Path
import hashlib
import json
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from bci_vla.scene.state import GroundedScene
from bci_vla.scene.tasks import BackendCapabilities
from bci_vla.models.consequence import OutcomeTable, ObservedTrajectory
from bci_vla.protocols import ProtocolError

def load(p):return json.loads((root/p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

p=load('execution/progress.json');t=load('execution/07_test_result.json')
check('01-07 remain complete',all(s['status']=='已完成' for s in p['steps'][:7]))
check('later progress preserved',int(p['current_step'])>=8)
check('real outcome calibration remains pending',p['steps'][6]['execution_scope']=='offline_software_and_simulation_branch' and p['steps'][6]['real_calibration']['status']=='待外部条件')
check('G1 scope unchanged and G2-G4 unstarted',p['research_gates']['G1']['scope']=='conditional_theory_and_protocol_design' and all(p['research_gates'][g]['status']=='未开始' for g in ['G2','G3','G4']))
check('66 software tests pass without skips',t['result']=='PASS' and t['tests_run']==66 and t['failures']==t['errors']==t['skipped']==0)
reuse=load('execution/07_reuse_manifest.json')
check('three original grammar files unchanged and exact snapshots',len(reuse['files'])==3 and all(hashlib.sha256(Path(f['source']).read_bytes()).hexdigest()==hashlib.sha256(Path(f['snapshot']).read_bytes()).hexdigest()==f['sha256'] for f in reuse['files']))
GroundedScene(load('examples/scene_outcomes/scene.json'));ObservedTrajectory(load('examples/scene_outcomes/trajectory.json'));BackendCapabilities(load('configs/consequences/simulation_backend.json'));OutcomeTable(load('configs/consequences/simulation_outcome_table.json'))
check('offline scene trajectory capability and outcome examples validate',True)
real=load('configs/consequences/real_outcome_uncalibrated.template.json')
check('real cost and checkpoint not fabricated',real['status']=='uncalibrated' and real['rows'] is None and real['weights'] is None and real['checkpoint_ref'] is None)
try:OutcomeTable(real)
except ProtocolError as e:check('real consequence loader blocked',str(e)=='real_consequence_calibration_unavailable')
else:raise AssertionError('real cost admitted')
a=(root/'execution/07_acceptance.txt').read_text(encoding='utf-8-sig')
check('all 07.1-07.9 acceptance entries',all(f'07.{i}' in a for i in range(1,10)))
check('frozen input and software hashes unchanged',all(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256'] for f in load('execution/07_source_manifest.json')['files']))
check('TXT encoding intact',all(q.read_bytes().startswith(b'\xef\xbb\xbf') and b'\n' not in q.read_bytes().replace(b'\r\n',b'') for q in root.rglob('*.txt')))
v={'scope':'offline_scene_and_outcome_delivery','result':'PASS','count':len(checks),'checks':checks,'real_consequence_calibration':'待外部条件','not_evaluated':['real object tracking','real backend capability','physical endpoint or recovery labels','real cost weights','G2-G4']}
out=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
output_name='07_integrity_result.json' if int(p['current_step'])==8 else '07_integrity_recheck_after_step08.json'
(root/'execution'/output_name).write_text(out,encoding='utf-8')
print(json.dumps(v,ensure_ascii=True,indent=2))
