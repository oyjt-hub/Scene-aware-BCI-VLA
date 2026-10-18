"""CPU interface/structural tests; never training or reasoning-performance claims."""
import copy,json,subprocess,sys,unittest
from pathlib import Path
import torch
from bci_vla.neural_interaction_v4.config import ModuleConfig
from bci_vla.neural_interaction_v4.tensorizer import collate_public
from bci_vla.neural_interaction_v4.model import matched_models,count_parameters
from bci_vla.neural_interaction_v4.losses import auxiliary_losses
from bci_vla.neural_interaction_v4.export import inspect_prediction
ROOT=Path(__file__).resolve().parents[1]
def clone(x):return {k:v.clone() for k,v in x.items()}

class ConnectorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.manual_seed(17);torch.set_num_threads(2)
  cls.rows=json.loads((ROOT/'execution/v4/06_fixtures/public.json').read_text())
  cls.labels=json.loads((ROOT/'execution/v4/06_fixtures/private.json').read_text())
  cls.models=matched_models()
  for model in cls.models.values():model.eval()
  cls.p=next(p for p in cls.rows if p['public_state']['diagnostics']['status']=='SUFFICIENT' and not p['public_state']['pending_question'])
  cls.x,cls.meta=collate_public([cls.p]);cls.full=cls.models['Full']
 def eq(self,a,b):torch.testing.assert_close(a,b,rtol=2e-5,atol=3e-6)
 def predict(self,x=None,model=None):
  with torch.no_grad():return (model or self.full)(x or self.x,return_trace=True)
 def test_01_actual_LIBERO_shapes_all_controls(self):
  self.assertTrue(all(p['provenance']['vision']=='privileged_simulator' for p in self.rows))
  self.assertGreaterEqual(len({len(p['scene']['objects']) for p in self.rows}),2)
  for start in range(0,len(self.rows),8):
   x,m=collate_public(self.rows[start:start+8])
   for name,model in self.models.items():
    y=self.predict(x,model);self.assertEqual(tuple(y['prefix'].shape),(len(m),8,2048))
    for k,v in y.items():
     if isinstance(v,torch.Tensor):self.assertTrue(torch.isfinite(v).all(),(name,k))
    self.eq(y['candidate_probabilities'].sum(1)+y['outside_probability'],torch.ones(len(m)))
    self.assertEqual(float(y['candidate_probabilities'][~x['candidate_mask']].abs().sum()),0.)
    self.assertFalse(y['execution_authorized']);self.assertFalse(y['model_trained'])
 def test_02_actual_parameter_budget(self):
  counts={k:count_parameters(v) for k,v in self.models.items()}
  for value in counts.values():self.assertLess(abs(value-counts['Full'])/counts['Full'],.05)
  self.assertFalse(any(isinstance(m,torch.nn.MultiheadAttention) for m in self.models['B3_aux'].modules()))
 def test_03_empty_observation_and_zero_N_H(self):
  p=json.loads((ROOT/'examples/v4/empty_scene_and_history.json').read_text())
  c=ModuleConfig(roi_dim=4,semantic_dim=3)
  # Same total feature dimension, no invented scene or BCI evidence.
  x,_=collate_public([p],c);model=self.full
  for k in ['objects','selection','candidate_mask']:x[k]=x[k][:,:0]
  for group in ['option','event','task','state','pending']:
   for side in ['src','dst']:x[group+'_'+side]=x[group+'_'+side][...,:0]
  for k in ['option_src','option_dst','option_numeric','option_mask','event_src','event_dst','event_numeric','event_mask','diag_history','diag_mask']:x[k]=x[k][:,:0]
  y=self.predict(x,model);self.eq(y['outside_probability'],torch.ones(1));self.assertFalse(bool(y['reliability_valid'][0]))
  self.assertEqual(y['candidate_probabilities'].shape[1],0)
 def test_04_batch_isolation(self):
  x,_=collate_public([self.p,self.rows[0],self.rows[-1]])
  for model in self.models.values():
   a=self.predict(model=model);b=self.predict(x,model)
   for key in ['prefix','candidate_probabilities','reliability_logit']:self.eq(a[key][0],b[key][0])
 def test_05_joint_object_permutation(self):
  p=copy.deepcopy(self.p);n=len(p['scene']['objects']);perm=list(reversed(range(n)))
  p['scene']['objects']=[p['scene']['objects'][i] for i in perm];x,m=collate_public([p])
  for model in self.models.values():
   a=self.predict(model=model);b=self.predict(x,model)
   self.eq(a['prefix'],b['prefix']);self.eq(a['candidate_probabilities'][0,perm],b['candidate_probabilities'][0,:n]);self.eq(a['outside_probability'],b['outside_probability'])
  self.assertEqual(m[0]['instance_ids'],[self.meta[0]['instance_ids'][i] for i in perm])
 def test_06_alias_label_not_numeric_features(self):
  p=copy.deepcopy(self.p)
  for i,o in enumerate(p['scene']['objects']):o['alias']='OBJ_'+str(15-i);o['label']='arbitrary semantic label'
  x,_=collate_public([p])
  for key in x:self.assertTrue(torch.equal(x[key],self.x[key]),key)
 def test_07_padding_NaN_ignored_valid_illegal_rejected(self):
  x=clone(self.x);cm=x['candidate_mask']
  x['objects'][~cm]=float('nan');x['selection'][~cm]=float('inf')
  x['diag_history'][~x['diag_mask']]=float('nan');x['event_numeric'][~x['event_mask']]=float('nan')
  x['option_numeric'][~x['option_mask']]=float('inf')
  for model in self.models.values():self.eq(self.predict(model=model)['prefix'],self.predict(x,model)['prefix'])
  for value in [float('nan'),float('inf'),1e5]:
   x=clone(self.x);x['objects'][0,0,0]=value
   with self.assertRaises(ValueError):self.full(x)
  x=clone(self.x);x['hidden_truth']=torch.zeros(1)
  with self.assertRaises(ValueError):self.full(x)
  x=clone(self.x);x['objects']=x['objects'][...,:-1]
  with self.assertRaises(ValueError):self.full(x)
 def test_08_channel_barrier_at_every_layer(self):
  x=clone(self.x);x['objects'][0,0,:6]+=.25;x['selection'][0,0,0]-=.3
  x['state_numeric'][0,2,9]=1.;x['event_numeric'][0,0,13]+=.1
  a=self.predict();b=self.predict(x)
  self.eq(a['trace']['diagnostic_history'],b['trace']['diagnostic_history'])
  for v,w in zip(a['trace']['channel_layers'],b['trace']['channel_layers']):self.eq(v,w)
  self.eq(a['trace']['typed_summary_tokens'],b['trace']['typed_summary_tokens']);self.eq(a['prefix'][:,4:],b['prefix'][:,4:]);self.eq(a['reliability_logit'],b['reliability_logit'])
  self.assertGreater(float((a['prefix'][:,:4]-b['prefix'][:,:4]).abs().max()),1e-6)
 def test_09_PROBE_object_semantics_invariance(self):
  p=copy.deepcopy(self.p);ids=[o['instance_id'] for o in p['scene']['objects']];changed=0
  collections=[p['history'],p['public_state']['diagnostics']['receipts']]
  for events in collections:
   for event in events:
    q=event.get('question')
    if q and q['role']=='PROBE':
     for opt in q['options']:
      if opt['reference'] in ids:opt['reference']=ids[(ids.index(opt['reference'])+1)%len(ids)];changed+=1
  self.assertGreater(changed,0)
  x,_=collate_public([p]);a=self.predict();b=self.predict(x)
  for key in x:self.assertTrue(torch.equal(x[key],self.x[key]),key)
  for key in ['prefix','candidate_probabilities','reliability_logit']:self.eq(a[key],b[key])
  for branch in ['channel_layers','intent_layers']:
   for u,v in zip(a['trace'][branch],b['trace'][branch]):self.eq(u,v)
  for branch in ['task_history','diagnostic_history','typed_summary_tokens']:self.eq(a['trace'][branch],b['trace'][branch])
 def test_10_diagnostic_numeric_effect_and_oneway_gradient(self):
  x=clone(self.x);self.assertTrue(bool(x['diag_mask'].any()));x['diag_history'][0,0,6]+=.4
  a=self.predict();b=self.predict(x)
  for key in ['prefix','reliability_logit']:self.assertGreater(float((a[key]-b[key]).abs().max()),1e-7)
  for u,v in zip(a['trace']['channel_layers'],b['trace']['channel_layers']):self.assertGreater(float((u-v).abs().max()),1e-7)
  x=clone(self.x);x['objects'].requires_grad_();x['event_numeric'].requires_grad_();x['diag_history'].requires_grad_()
  y=self.full(x);grads=torch.autograd.grad(y['reliability_logit'].sum(),[x['objects'],x['event_numeric'],x['diag_history']],allow_unused=True)
  self.assertIsNone(grads[0]);self.assertIsNone(grads[1]);self.assertGreater(float(grads[2].abs().sum()),0.)
 def test_11_persistent_summary_or_insufficient(self):
  x=clone(self.x);x['diag_mask'].zero_();x['diag_history'].zero_()
  y=self.predict(x);self.assertTrue(bool(y['reliability_valid'][0]))
  x['diag_summary'].zero_();x['diag_summary'][:,4]=1.;x['diag_summary'][:,9]=1.
  y=self.predict(x);self.assertFalse(bool(y['reliability_valid'][0]));self.assertEqual(float(y['reliability_probability'][0]),0.)
  self.assertIsNone(inspect_prediction(y,self.meta)['channel_raw_probability'])
 def test_12_long_probe_history_preserves_confirmed_task(self):
  p=json.loads((ROOT/'examples/v4/confirmed_task_survives_nine_probes.json').read_text())
  self.assertEqual(p['public_state']['diagnostics']['attempt_count'],9);self.assertLessEqual(len(p['history']),8)
  task=copy.deepcopy(p['public_state']['confirmed_task']);x,_=collate_public([p],ModuleConfig(roi_dim=4,semantic_dim=3))
  self.assertTrue(bool(x['state_mask'][0,0]));self.assertEqual(float(x['state_src'][0,0].sum()),1.)
  self.predict(x);self.assertEqual(task,p['public_state']['confirmed_task'])
  p['history'].append(copy.deepcopy(p['history'][-1]))
  with self.assertRaises(ValueError):collate_public([p],ModuleConfig(roi_dim=4,semantic_dim=3))
 def test_13_STOP_CANCEL_no_torch_import(self):
  code="import sys,json;from pathlib import Path;from bci_vla.neural_interaction_v4.control import route_control;assert 'torch' not in sys.modules;"+"assert not route_control(json.loads(Path('examples/v4/stop_clears_task.json').read_text()))['call_neural_module'];assert not route_control(json.loads(Path('examples/v4/cancel_clears_task.json').read_text()))['call_neural_module']"
  r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_14_private_loss_binding_no_forward_label(self):
  label=next(v for v in self.labels if v['sample_id']==self.p['sample_id']);y=self.full(self.x)
  losses=auxiliary_losses(y,[self.p],[label],self.meta);self.assertTrue(torch.isfinite(losses['total']))
  # Frozen conservative all+outside SET deliberately contributes ~zero information.
  self.assertLess(abs(float(losses['intent'].detach())),1e-5)
  bad=copy.deepcopy(label);bad['public_input_sha256']='0'*64
  with self.assertRaises(ValueError):auxiliary_losses(y,[self.p],[bad],self.meta)
  labels=[v for v in self.labels if v['sample_id']==self.p['sample_id']]
  for label in labels:auxiliary_losses(y,[self.p],[label],self.meta)
  pair=next(p for p in self.rows if sum(v['sample_id']==p['sample_id'] for v in self.labels)>1)
  variants=[v for v in self.labels if v['sample_id']==pair['sample_id']];self.assertNotEqual(variants[0]['hidden_truth'],variants[1]['hidden_truth'])
  x,m=collate_public([pair]);a=self.predict(x)
  for variant in variants:auxiliary_losses(a,[pair],[variant],m)
  self.eq(a['prefix'],self.predict(x)['prefix'])
 def test_15_backward_all_models_and_components(self):
  for name,model in self.models.items():
   model.zero_grad(set_to_none=True);y=model(self.x)
   label=next(v for v in self.labels if v['sample_id']==self.p['sample_id'])
   losses=auxiliary_losses(y,[self.p],[label],self.meta)
   # Explicit mechanism-only objective probes otherwise unsupervised output wiring.
   objective=losses['total']+.01*y['prefix'].square().mean()+.01*y['missing_slot_logits'].square().mean()+.01*y['abstain_logit'].square().mean()
   objective.backward();groups={}
   for key,p in model.named_parameters():
    if p.grad is not None:
     self.assertTrue(torch.isfinite(p.grad).all(),(name,key));groups[key.split('.')[0]]=groups.get(key.split('.')[0],0.)+float(p.grad.abs().sum())
   for group in ['features','reliability','typed_summary','prefix_projection','pointer_object','pointer_intent','missing_slots','abstain','interaction']:
    # all-candidate SET has no discriminative pointer gradient; measured separately below
    if group not in ['pointer_object','pointer_intent']:self.assertGreater(groups.get(group,0.),0.,(name,group))
   if name=='Full':
    for group in ['task_history','diagnostic_history','intent_queries','channel_queries','intent_blocks','channel_blocks']:self.assertGreater(groups.get(group,0.),0.,group)
   model.zero_grad(set_to_none=True)
   y=model(self.x);(-y['candidate_logits'][0,0]+y['outside_probability'].sum()).backward()
   for group in ['pointer_object','pointer_intent','outside']:self.assertGreater(sum(float(p.grad.abs().sum()) for k,p in model.named_parameters() if k.startswith(group) and p.grad is not None),0.)
 def test_16_public_hidden_field_and_feature_profile_rejected(self):
  p=copy.deepcopy(self.p);p['hidden_truth']={}
  with self.assertRaises(ValueError):collate_public([p])
  with self.assertRaises(ValueError):collate_public([self.p],ModuleConfig(roi_dim=5))
 def test_17_option_order_invariance(self):
  p=copy.deepcopy(self.p)
  for e in p['history']:
   if e['question']:e['question']['options'].reverse();e['evidence']['values'].reverse()
  for e in p['public_state']['diagnostics']['receipts']:
   e['question']['options'].reverse();e['evidence']['values'].reverse()
  x,_=collate_public([p]);self.eq(self.predict()['prefix'],self.predict(x)['prefix'])
 def test_18_object_gradient_equivariance(self):
  p=copy.deepcopy(self.p);n=len(p['scene']['objects']);perm=list(reversed(range(n)));p['scene']['objects'].reverse()
  x,_=collate_public([p]);a=clone(self.x);b=clone(x);a['objects'].requires_grad_();b['objects'].requires_grad_()
  u=self.full(a)['prefix'].square().mean();v=self.full(b)['prefix'].square().mean()
  ga=torch.autograd.grad(u,a['objects'])[0];gb=torch.autograd.grad(v,b['objects'])[0]
  self.eq(ga[0,perm],gb[0,:n]);self.assertGreater(float(ga.abs().sum()),0.)
 def test_19_tensor_history_bound_and_pending_mask(self):
  x=clone(self.x)
  for k in ['option_src','option_dst','option_numeric','option_mask','event_src','event_dst','event_numeric','event_mask','diag_history','diag_mask']:
   shape=list(x[k].shape);shape[1]=1;x[k]=torch.cat([x[k],torch.zeros(shape,dtype=x[k].dtype)],1)
  with self.assertRaises(ValueError):self.full(x)
  x=clone(self.x);x['pending'][:]=True;y=self.predict(x)
  self.assertEqual(int(y['interaction_logits'][0].argmax()),3);self.assertFalse(y['execution_authorized'])
 def test_20_shared_controls_do_not_have_Full_barrier(self):
  x=clone(self.x);x['objects'][0,0,:6]+=.25;x['selection'][0,0,0]-=.3
  for name in ['B3_aux','B4_aux']:
   a=self.predict(model=self.models[name]);b=self.predict(x,self.models[name])
   self.assertGreater(float((a['reliability_logit']-b['reliability_logit']).abs().max()),1e-7)

if __name__=='__main__':unittest.main(verbosity=2)
