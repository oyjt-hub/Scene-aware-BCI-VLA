"""Small deterministic development-only LIBERO CPU mechanism fixture bundle."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'execution/v4/06_fixtures'
def main():
 OUT.mkdir(exist_ok=True);publics=[];labels=[];sources=[]
 for split in ['train','validation']:
  pp=ROOT/f'data/v4/{split}/public_input.jsonl';lp=pp.with_name('private_labels.jsonl')
  for p in [pp,lp]:sources.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  rows=[json.loads(s) for s in pp.read_text().splitlines()];labs=[json.loads(s) for s in lp.read_text().splitlines()];chosen=[];seen=set()
  # Cover candidate counts, support/expiry/confirmation/pending states and task roles.
  for p in rows:
   state=p['public_state'];key=(len(p['scene']['objects']),state['diagnostics']['status'],state['confirmed_task'] is not None,state['pending_question'] is not None,tuple(sorted(set(e['role'] for e in p['history']))))
   if key not in seen:chosen.append(p);seen.add(key)
   if len(chosen)>=32:break
  # Include a real frozen identical-public / different-private counterfactual pair.
  from collections import Counter
  repeated={sid for sid,count in Counter(v['sample_id'] for v in labs).items() if count>1}
  counterfactual=next(p for p in rows if p['sample_id'] in repeated)
  if counterfactual['sample_id'] not in {p['sample_id'] for p in chosen}:chosen.append(counterfactual)
  ids={p['sample_id'] for p in chosen};publics.extend(chosen);labels.extend(p for p in labs if p['sample_id'] in ids)
 (OUT/'public.json').write_text(json.dumps(publics)+'\n',encoding='utf-8')
 (OUT/'private.json').write_text(json.dumps(labels)+'\n',encoding='utf-8')
 report={'platform':'actual frozen LIBERO captures; simulated BCI','splits':['train','validation'],'public_count':len(publics),'private_count':len(labels),'sources':sources,'formal_test_accessed':False,'calibration_accessed':False}
 (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
