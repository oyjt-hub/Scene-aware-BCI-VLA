"""Private supervision is bound and consumed here, never passed to forward.

No optimizer or training loop. SET labels penalize mass outside the supported set;
the current LIBERO conservative all-candidate SET has zero information by design.
"""
import torch
from torch.nn import functional as F
from bci_vla.data_v4.contracts import validate_private
from bci_vla.interaction_contract_v4.validation import canonical_digest

ACTIONS=['PROPOSE_TASK','ASK_INTENT','CHECK_CHANNEL','WAIT']

def auxiliary_losses(output, publics, labels, metadata):
 if not (len(publics)==len(labels)==len(metadata)==output['prefix'].shape[0]):raise ValueError('batch binding')
 zero=output['prefix'].sum()*0.;terms={'intent':[], 'channel':[], 'interaction':[]}
 for i,(p,label,meta) in enumerate(zip(publics,labels,metadata)):
  validate_private(label,p)
  if meta['input_sha256']!=canonical_digest(p):raise ValueError('public tensor binding mismatch')
  target=label['labels'];intent=target['intent']
  if intent['mask']:
   ids=meta['instance_ids'];allowed=[ids.index(v) for v in intent['candidate_ids']]
   if intent['outside']:allowed.append(output['candidate_logits'].shape[1]-1)
   if not allowed:raise ValueError('empty supported intent set')
   logp=F.log_softmax(output['candidate_logits'][i],dim=-1)
   terms['intent'].append(-torch.logsumexp(logp[allowed],dim=0))
  channel=target['channel']
  if channel['mask']:
   if not bool(output['reliability_valid'][i]):raise ValueError('channel label without public diagnostic support')
   y=output['reliability_logit'].new_tensor(float(channel['next_probe_success']))
   terms['channel'].append(F.binary_cross_entropy_with_logits(output['reliability_logit'][i],y))
  interaction=target['interaction']
  if interaction['mask']:
   y=torch.tensor([ACTIONS.index(interaction['action'])],device=output['prefix'].device)
   terms['interaction'].append(F.cross_entropy(output['interaction_logits'][i:i+1],y))
 result={key:torch.stack(values).mean() if values else zero for key,values in terms.items()}
 result['total']=sum(result.values());result['supervised_counts']={k:len(v) for k,v in terms.items()}
 return result
