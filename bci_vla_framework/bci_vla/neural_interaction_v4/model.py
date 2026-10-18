"""Untrained CPU connectors. Full has a structural diagnostic information barrier."""
import math
import torch
from torch import nn
from .config import ModuleConfig,EVENT_DIM,OPTION_DIM,STATE_DIM,DIAG_DIM,SUMMARY_DIM

FLOAT_KEYS={'objects','selection','option_src','option_dst','option_numeric','event_src','event_dst','event_numeric','task_src','task_dst','task_numeric','state_src','state_dst','state_numeric','pending_src','pending_dst','pending_numeric','diag_history','diag_summary'}
BOOL_KEYS={'candidate_mask','option_mask','event_mask','task_mask','state_mask','pending_mask','diag_mask','stop','pending'}

def clean_batch(batch,c):
 if set(batch)!=FLOAT_KEYS|BOOL_KEYS:raise ValueError('forward accepts only allowlisted tensors, never metadata/private labels')
 if any(not isinstance(v,torch.Tensor) for v in batch.values()):raise ValueError('tensor inputs required')
 if batch['candidate_mask'].ndim!=2 or batch['event_mask'].ndim!=2:raise ValueError('mask rank')
 b,n=batch['candidate_mask'].shape;h=batch['event_mask'].shape[1]
 if b<1 or n>16 or h>8:raise ValueError('unsupported B/N/H')
 shapes={'objects':(b,n,c.roi_dim+c.semantic_dim+6),'selection':(b,n,7),'candidate_mask':(b,n),
  'option_src':(b,h,16,n),'option_dst':(b,h,16,n),'option_numeric':(b,h,16,OPTION_DIM),'option_mask':(b,h,16),
  'event_src':(b,h,n),'event_dst':(b,h,n),'event_numeric':(b,h,EVENT_DIM),'event_mask':(b,h),
  'task_src':(b,32,n),'task_dst':(b,32,n),'task_numeric':(b,32,OPTION_DIM),'task_mask':(b,32),
  'state_src':(b,3,n),'state_dst':(b,3,n),'state_numeric':(b,3,STATE_DIM),'state_mask':(b,3),
  'pending_src':(b,16,n),'pending_dst':(b,16,n),'pending_numeric':(b,16,OPTION_DIM),'pending_mask':(b,16),
  'diag_history':(b,h,DIAG_DIM),'diag_mask':(b,h),'diag_summary':(b,SUMMARY_DIM),'stop':(b,),'pending':(b,)}
 for key,v in batch.items():
  if tuple(v.shape)!=shapes[key] or v.dtype!=(torch.bool if key in BOOL_KEYS else torch.float32):raise ValueError('tensor shape/dtype: '+key)
  if v.device!=batch['objects'].device:raise ValueError('mixed tensor devices')
 cm=batch['candidate_mask'];em=batch['event_mask'];om=batch['option_mask'];dm=batch['diag_mask']
 masks={'objects':cm[...,None],'selection':cm[...,None],
  'option_src':em[:,:,None,None]&om[:,:,:,None]&cm[:,None,None,:],'option_dst':em[:,:,None,None]&om[:,:,:,None]&cm[:,None,None,:],
  'option_numeric':em[:,:,None,None]&om[:,:,:,None], 'event_src':em[:,:,None]&cm[:,None,:],'event_dst':em[:,:,None]&cm[:,None,:],'event_numeric':em[:,:,None],
  'diag_history':dm[:,:,None], 'diag_summary':torch.ones_like(batch['diag_summary'],dtype=torch.bool)}
 for name in ['task','state','pending']:
  masks[name+'_src']=batch[name+'_mask'][:,:,None]&cm[:,None,:];masks[name+'_dst']=masks[name+'_src'];masks[name+'_numeric']=batch[name+'_mask'][:,:,None]
 result=dict(batch)
 for key in FLOAT_KEYS:
  v=torch.where(masks[key],batch[key],torch.zeros_like(batch[key]))
  if not torch.isfinite(v).all() or (v.abs()>1e4).any():raise ValueError('nonfinite/out-of-profile valid tensor: '+key)
  result[key]=v
 return result

def masked_mean(x,mask,dim=1):
 return (x*mask.unsqueeze(-1)).sum(dim)/mask.sum(dim).clamp_min(1).unsqueeze(-1)

class CommonFeatures(nn.Module):
 def __init__(self,c):
  super().__init__();d=c.hidden_dim
  def project(i):return nn.Sequential(nn.Linear(i,d),nn.GELU(),nn.LayerNorm(d))
  self.object=project(c.roi_dim+c.semantic_dim+13)
  self.option=nn.Sequential(nn.Linear(2*d+OPTION_DIM,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
  self.event=project(3*d+EVENT_DIM);self.state=project(2*d+STATE_DIM);self.diagnostic=project(DIAG_DIM);self.summary=project(SUMMARY_DIM)
  self.empty=nn.ParameterDict({k:nn.Parameter(torch.randn(1,1,d)*.02) for k in ['objects','events','diagnostic','states','tasks','pending']})
 def option_tokens(self,obj,src,dst,num):
  a=torch.einsum('b...n,bnd->b...d',src,obj);z=torch.einsum('b...n,bnd->b...d',dst,obj)
  return self.option(torch.cat([a,z,num],dim=-1))
 def append_empty(self,name,x,mask):
  empty=self.empty[name].expand(x.shape[0],-1,-1)
  return torch.cat([x,empty],1),torch.cat([mask,~mask.any(1,keepdim=True)],1)
 def forward(self,x):
  obj=self.object(torch.cat([x['objects'],x['selection']],-1))*x['candidate_mask'][...,None]
  opt=self.option_tokens(obj,x['option_src'],x['option_dst'],x['option_numeric'])
  opt=masked_mean(opt,x['option_mask'],dim=2)
  source=torch.einsum('bhn,bnd->bhd',x['event_src'],obj);dest=torch.einsum('bhn,bnd->bhd',x['event_dst'],obj)
  event=self.event(torch.cat([opt,source,dest,x['event_numeric']],-1))*x['event_mask'][...,None]
  source=torch.einsum('bsn,bnd->bsd',x['state_src'],obj);dest=torch.einsum('bsn,bnd->bsd',x['state_dst'],obj)
  state=self.state(torch.cat([source,dest,x['state_numeric']],-1))*x['state_mask'][...,None]
  task=self.option_tokens(obj,x['task_src'],x['task_dst'],x['task_numeric'])*x['task_mask'][...,None]
  pending=self.option_tokens(obj,x['pending_src'],x['pending_dst'],x['pending_numeric'])*x['pending_mask'][...,None]
  diagnostic=self.diagnostic(x['diag_history'])*x['diag_mask'][...,None]
  groups={name:self.append_empty(name,tokens,mask) for name,tokens,mask in [('objects',obj,x['candidate_mask']),('events',event,x['event_mask']),('diagnostic',diagnostic,x['diag_mask']),('states',state,x['state_mask']),('tasks',task,x['task_mask']),('pending',pending,x['pending_mask'])]}
  summary=self.summary(x['diag_summary'])[:,None,:]
  groups['summary']=(summary,torch.ones(summary.shape[:2],device=summary.device,dtype=torch.bool))
  return obj,groups

class CrossBlock(nn.Module):
 def __init__(self,c):
  super().__init__();d=c.hidden_dim
  self.self_attn=nn.MultiheadAttention(d,c.heads,batch_first=True,dropout=0.)
  self.cross=nn.MultiheadAttention(d,c.heads,batch_first=True,dropout=0.)
  self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
  self.ffn=nn.Sequential(nn.Linear(d,c.ffn_dim),nn.GELU(),nn.Linear(c.ffn_dim,d))
 def forward(self,q,memory,valid):
  z=self.norms[0](q);q=q+self.self_attn(z,z,z,need_weights=False)[0]
  z=self.norms[1](q);q=q+self.cross(z,memory,memory,key_padding_mask=~valid,need_weights=False)[0]
  return q+self.ffn(self.norms[2](q))

def encoder(c):return nn.TransformerEncoderLayer(c.hidden_dim,c.heads,c.ffn_dim,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
def concatenate(groups,names):return torch.cat([groups[k][0] for k in names],1),torch.cat([groups[k][1] for k in names],1)

class BaseConnector(nn.Module):
 def __init__(self,c=None):
  super().__init__();self.config=c or ModuleConfig();d=self.config.hidden_dim
  self.features=CommonFeatures(self.config)
  self.reliability=nn.Linear(d,1)
  self.typed_summary=nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.Linear(d,4*d))
  self.prefix_projection=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,self.config.vlm_dim))
  self.pointer_object=nn.Linear(d,d);self.pointer_intent=nn.Linear(d,d)
  self.outside=nn.Linear(d,1);self.missing_slots=nn.Linear(d,3);self.abstain=nn.Linear(d,1);self.interaction=nn.Linear(d,4)
 def summarize(self,x,diag_latent):
  logit=self.reliability(diag_latent).squeeze(-1);s=x['diag_summary']
  support=(s[:,0]>=3/64)&(s[:,3]>=2/16)&(s[:,4]<=.5)&(s[:,5]==0)&(s[:,6]==1)&(s[:,8]==0)
  if not torch.equal(support,s[:,7].bool()):raise ValueError('inconsistent public diagnostic support')
  probability=torch.where(support,logit.sigmoid(),torch.zeros_like(logit))
  evidence=x['diag_mask'].any(1)|(s[:,6].bool()&~s[:,8].bool())
  typed=torch.cat([s,probability[:,None],evidence.float()[:,None]],-1)
  return logit,probability,support,self.typed_summary(typed).reshape(-1,4,self.config.hidden_dim)
 def finish(self,x,obj,intent,prefix_latent,channel,trace):
  logit,probability,support,summary_tokens=channel;pooled=intent.mean(1)
  score=torch.einsum('bnd,bd->bn',self.pointer_object(obj),self.pointer_intent(pooled))/math.sqrt(self.config.hidden_dim)
  score=score.masked_fill(~x['candidate_mask'],-1e9)
  logits=torch.cat([score,self.outside(pooled)],-1);distribution=logits.softmax(-1)
  interaction=self.interaction(pooled);wait=x['stop']|x['pending']
  wait_logits=torch.full_like(interaction,-1e9);wait_logits[:,3]=0.
  interaction=torch.where(wait[:,None],wait_logits,interaction)
  return {'prefix':self.prefix_projection(prefix_latent),'candidate_logits':logits,'candidate_probabilities':distribution[:,:-1], 'outside_probability':distribution[:,-1],
   'missing_slot_logits':self.missing_slots(pooled),'abstain_logit':self.abstain(pooled).squeeze(-1),'interaction_logits':interaction,
   'reliability_logit':logit,'reliability_probability':probability,'reliability_valid':support,'diagnostic_expired':x['diag_summary'][:,8].bool(),
   'probability_calibrated':False,'execution_authorized':False,'model_trained':False,'trace':trace}

class IntentChannelConnector(BaseConnector):
 """Full: no object/task input path exists into the diagnostic history/queries."""
 def __init__(self,c=None):
  super().__init__(c);c=self.config;d=c.hidden_dim
  self.task_history=encoder(c);self.diagnostic_history=encoder(c)
  self.intent_queries=nn.Parameter(torch.randn(1,4,d)*.02);self.channel_queries=nn.Parameter(torch.randn(1,4,d)*.02)
  self.intent_blocks=nn.ModuleList([CrossBlock(c) for _ in range(c.fusion_blocks)])
  self.channel_blocks=nn.ModuleList([CrossBlock(c) for _ in range(c.fusion_blocks)])
 def forward(self,batch,return_trace=False):
  x=clean_batch(batch,self.config);obj,g=self.features(x)
  task,tm=g['events'];diag,dm=g['diagnostic']
  task=self.task_history(task,src_key_padding_mask=~tm)*tm[...,None]
  diag=self.diagnostic_history(diag,src_key_padding_mask=~dm)*dm[...,None]
  g['events']=(task,tm);g['diagnostic']=(diag,dm)
  memory,valid=concatenate(g,['diagnostic','summary']);q=self.channel_queries.expand(obj.shape[0],-1,-1);channel_layers=[]
  for block in self.channel_blocks:q=block(q,memory,valid);channel_layers.append(q)
  channel=self.summarize(x,q.mean(1));summary_tokens=channel[3]
  memory,valid=concatenate(g,['objects','events','states','tasks','pending'])
  memory=torch.cat([memory,summary_tokens],1);valid=torch.cat([valid,torch.ones(summary_tokens.shape[:2],dtype=torch.bool,device=valid.device)],1)
  q=self.intent_queries.expand(obj.shape[0],-1,-1);intent_layers=[]
  for block in self.intent_blocks:q=block(q,memory,valid);intent_layers.append(q)
  trace={'task_history':task,'diagnostic_history':diag,'channel_layers':channel_layers,'typed_summary_tokens':summary_tokens,'intent_layers':intent_layers} if return_trace else {}
  return self.finish(x,obj,q,torch.cat([q,summary_tokens],1),channel,trace)

class SharedQueryConnector(BaseConnector):
 """B4: ordinary shared history and 8 shared queries, same total block budget."""
 def __init__(self,c=None):
  super().__init__(c);c=self.config
  self.shared_history=nn.ModuleList([encoder(c),encoder(c)])
  self.queries=nn.Parameter(torch.randn(1,8,c.hidden_dim)*.02)
  self.blocks=nn.ModuleList([CrossBlock(c) for _ in range(2*c.fusion_blocks)])
 def forward(self,batch,return_trace=False):
  x=clean_batch(batch,self.config);obj,g=self.features(x);memory,valid=concatenate(g,list(g))
  for layer in self.shared_history:memory=layer(memory,src_key_padding_mask=~valid)*valid[...,None]
  q=self.queries.expand(obj.shape[0],-1,-1);layers=[]
  for block in self.blocks:q=block(q,memory,valid);layers.append(q)
  channel=self.summarize(x,q.mean(1));prefix=q+channel[3].mean(1,keepdim=True)
  return self.finish(x,obj,q,prefix,channel,{'shared_layers':layers} if return_trace else {})

class PooledMLPConnector(BaseConnector):
 """B3: mask-aware group pooling then a genuine MLP, without attention layers."""
 def __init__(self,c=None,width=1024):
  super().__init__(c);d=self.config.hidden_dim;self.width=width
  self.pool_mlp=nn.Sequential(nn.Linear(7*d,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,8*d))
 def forward(self,batch,return_trace=False):
  x=clean_batch(batch,self.config);obj,g=self.features(x)
  pooled=torch.cat([masked_mean(tokens,valid) for tokens,valid in g.values()],-1)
  q=self.pool_mlp(pooled).reshape(-1,8,self.config.hidden_dim);channel=self.summarize(x,q.mean(1))
  prefix=q+channel[3].mean(1,keepdim=True)
  return self.finish(x,obj,q,prefix,channel,{'pooled_latents':q} if return_trace else {})

def count_parameters(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)

def matched_models(config=None):
 c=config or ModuleConfig();full=IntentChannelConnector(c);shared=SharedQueryConnector(c)
 tiny=PooledMLPConnector(c,width=8);constant=count_parameters(tiny)-sum(p.numel() for p in tiny.pool_mlp.parameters());target=count_parameters(full)
 def mlp_parameters(w):return 2*w*w+(15*c.hidden_dim+3)*w+8*c.hidden_dim
 width=min(range(64,2049,8),key=lambda w:abs(constant+mlp_parameters(w)-target))
 pool=PooledMLPConnector(c,width=width)
 if max(abs(count_parameters(m)-target)/target for m in [shared,pool])>.05:raise ValueError('matched adapter budget not achieved')
 return {'Full':full,'B3_aux':pool,'B4_aux':shared}
