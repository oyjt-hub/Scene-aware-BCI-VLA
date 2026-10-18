"""Dataset integrity tests, not task-success or model-accuracy experiments."""
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from bci_vla.data_v4.channel import answer,question_key,world_at,digest
from bci_vla.data_v4.contracts import iter_public
from bci_vla.interaction_contract_v4.validation import _shape,_public,_private,canonical_digest,ContractError
DATA=ROOT/'data/v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def main():
 checks=[]
 def check(name,value):
  checks.append({'name':name,'passed':bool(value)})
 manifest=read(DATA/'data_manifest.json');split=read(DATA/'split_manifest.json');captures=read(DATA/'captures/capture_manifest.json')
 check('actual_LIBERO_48_base_episodes',captures['status']=='PASS' and len(captures['episodes'])==48)
 check('no_successful_task_claim',manifest['successful_task_demonstrations'] is False and not any(c['benchmark_goal_success'] for c in captures['episodes']))
 groups={p:{t['group_id'] for t in ts} for p,ts in split['partitions'].items()}
 for a in groups:
  for b in groups:
   if a<b:check('scene_disjoint:'+a+':'+b,not groups[a]&groups[b])
 lineage=rows(DATA/'lineage.jsonl');mapping={};image_owners={};state_owners={}
 for c in captures['episodes']:
  for table,key in [(state_owners,c['initial_state_sha256'])]:table.setdefault(key,set()).add(c['partition'])
  for frame in c['frames']:
   p=DATA/frame['image_path'];check('image_sha:'+p.name,hashlib.sha256(p.read_bytes()).hexdigest()==frame['image_sha256'])
   image_owners.setdefault(frame['image_sha256'],set()).add(c['partition'])
 check('initial_state_no_cross_partition',all(len(v)==1 for v in state_owners.values()))
 check('identical_image_no_cross_partition',all(len(v)==1 for v in image_owners.values()))
 for row in lineage:mapping.setdefault(row['base_episode_id'],set()).add(row['partition'])
 check('augmentations_grouped_by_base',all(len(v)==1 for v in mapping.values()) and len(mapping)==48)
 check('test_not_rendered',not (DATA/'test_sealed').exists() and not captures['test_rendered'] and not manifest['formal_test_generated'])
 check('test_reserved_8',sum(len(t['init_indices']) for t in split['partitions']['test_sealed'])==8)
 schema=read(ROOT/'configs/v4/interfaces_data_4_1.json');defs=schema['$defs'];public_all={};private_count=0;future_count=0;examples=[]
 for part in ['train','validation','calibration_fit','calibration_audit']:
  pubs=rows(DATA/part/'public_input.jsonl');privs=rows(DATA/part/'private_labels.jsonl');lookup={p['sample_id']:p for p in pubs}
  for p in pubs:_shape(p,defs['public_input'],defs);_public(p)
  for p in privs:_shape(p,defs['private_labels'],defs);_private(p,lookup[p['sample_id']]);private_count+=1
  for f in rows(DATA/part/'private_future_probe.jsonl'):
   p=lookup[f['sample_id']];q=f['evidence']['question'];symbol=next(o['symbol'] for o in q['options'] if o['option_id']==q['diagnostic_option_id'])
   assert symbol in p['public_state']['diagnostics']['covered_symbols']
   assert q['issued_at_s']>p['as_of_s'];future_count+=1
  check('schema_and_semantics:'+part,len(pubs)>0)
  public_all.update(lookup);examples.extend(pubs[:1])
 check('row_counts',len(public_all)==manifest['public_rows'] and private_count==manifest['private_label_rows'])
 check('no_partition_in_public',all('partition' not in p and 'split' not in p and 'hidden_truth' not in p and 'future' not in p for p in public_all.values()))
 # Latent goal variants share precisely the same public snapshot and supervised labels.
 for part in ['train','validation','calibration_fit','calibration_audit']:
  grouped={}
  for p in rows(DATA/part/'private_labels.jsonl'):grouped.setdefault(p['sample_id'],[]).append(p)
  pairs=[v for v in grouped.values() if len(v)==2]
  check('counterfactuals:'+part,all(a['public_input_sha256']==b['public_input_sha256'] and a['labels']==b['labels'] and a['hidden_truth']['goal']!=b['hidden_truth']['goal'] for a,b in pairs) and bool(pairs))
 public=examples[0];q=deepcopy(public['history'][0]['question'])
 world={'episode_id':'test-base','initial_goal':q['options'][0]['reference'],'later_goal':q['options'][1]['reference'],'initial_success':.9,'later_success':.35,'change_at_s':8.}
 response=answer(world,q,2.)
 altered=deepcopy(q);altered['question_id']='another-policy-question';altered['context']['session_id']='another-policy-session'
 check('noise_method_and_session_independent',answer(world,altered,2.)==response)
 answer(world,q,4.);answer(world,q,6.)
 check('noise_call_order_independent',answer(world,q,2.)==response)
 reversed_q=deepcopy(q);reversed_q['options'].reverse()
 check('noise_JSON_option_order_independent',answer(world,reversed_q,2.)==response)
 check('world_switch_is_time_not_questions',world_at(world,7.9)[0]!=world_at(world,8.)[0])
 for pair in [('initial_goal','later_goal'),('initial_success','later_success')]:check('world_has_explicit_schedule:'+pair[0],all(pair[0] in w and pair[1] in w and w['change_at_s']==8. for w in rows(DATA/'private_worlds.jsonl')))
 check('raw_scores_not_probabilities',all(e['evidence'] is None or e['evidence']['score_kind'] in ['RAW_SCORE','NONE'] for p in public_all.values() for e in p['history']))
 bad=deepcopy(public);bad['hidden_truth']={}
 try:_shape(bad,defs['public_input'],defs);rejected=False
 except ContractError:rejected=True
 check('reject_private_field_in_public',rejected)
 try:list(iter_public(DATA,'test_sealed'));rejected=False
 except ValueError:rejected=True
 check('reader_rejects_test',rejected)
 for row in read(ROOT/'execution/v4/04_before_manifest.json')['files']:check('preserved:'+row['path'],hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'])
 report={'scope':'actual LIBERO capture/dataset/lineage integrity only; not manipulation success','status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':len(checks),'passed':sum(c['passed'] for c in checks),'public_rows_validated':len(public_all),'private_rows_validated':private_count,'future_labels_checked':future_count,'results':checks}
 (ROOT/'execution/v4/04_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='results'}));print(json.dumps([c for c in checks if not c['passed']]))
 return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
