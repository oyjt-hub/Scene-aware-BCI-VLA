from dataclasses import dataclass

EVENT_DIM=17
OPTION_DIM=11
STATE_DIM=13
DIAG_DIM=40
SUMMARY_DIM=10

@dataclass(frozen=True)
class ModuleConfig:
 roi_dim:int=6
 semantic_dim:int=1
 hidden_dim:int=256
 heads:int=4
 fusion_blocks:int=2
 ffn_dim:int=1024
 vlm_dim:int=2048
 max_objects:int=16
 history_events:int=8
 task_options:int=32
 dropout:float=0.
 def __post_init__(self):
  if min(self.roi_dim,self.semantic_dim,self.hidden_dim,self.heads,self.ffn_dim,self.vlm_dim)<1 or self.hidden_dim%self.heads or self.max_objects!=16 or self.history_events!=8 or self.task_options!=32 or self.fusion_blocks<1 or self.dropout!=0:raise ValueError('unsupported prototype configuration')
