"""Small scoped LoRA without global dependencies; disabled is exact base call."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F

class HighLevelLoRALinear(nn.Module):
 def __init__(self,base,rank=16,alpha=16):
  super().__init__()
  if not isinstance(base,nn.Linear) or rank<1:raise ValueError('linear LoRA target/rank')
  self.base=base;self.base.requires_grad_(False);self.scale=alpha/rank;self.enabled=False
  self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
  self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):
  y=self.base(x)
  if not self.enabled:return y
  return y+(F.linear(F.linear(x.float(),self.A),self.B)*self.scale).to(y.dtype)

def install_high_level_lora(model,rank=16,alpha=16,targets=('q_proj','v_proj','o_proj')):
 """Scope strictly to PaliGemma language self-attention, never expert/vision/head."""
 layers=model.paligemma_with_expert.paligemma.language_model.layers;installed={}
 for i,layer in enumerate(layers):
  for name in targets:
   if name not in ['q_proj','k_proj','v_proj','o_proj']:raise ValueError('attention-only initial remediation')
   old=getattr(layer.self_attn,name)
   if isinstance(old,HighLevelLoRALinear):raise ValueError('already adapted')
   wrapped=HighLevelLoRALinear(old,rank,alpha);setattr(layer.self_attn,name,wrapped);installed[f'layers.{i}.self_attn.{name}']=wrapped
 return installed

@contextmanager
def high_level_scope(adapters,enabled=True):
 previous={k:v.enabled for k,v in adapters.items()}
 try:
  for v in adapters.values():v.enabled=enabled
  yield
 finally:
  for k,v in adapters.items():v.enabled=previous[k]

def adapter_state(adapters):
 return {f'{name}.{side}':getattr(module,side).detach().contiguous() for name,module in adapters.items() for side in ['A','B']}
