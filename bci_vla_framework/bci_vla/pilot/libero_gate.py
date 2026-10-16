"""Simulation-only task/chunk binding. This module has no robot transport."""
from collections import deque
import math
from ..protocols import require


class LiberoSimulationGate:
    def __init__(self):
        self.context=None;self.confirmed=None;self.queue=deque();self.stopped=False;self.mode='TASK_QUERY'
        self.discarded_actions=0;self.applied_actions=0

    def bind(self,*,suite,task_id,language,scene_revision):
        require(isinstance(suite,str) and bool(suite) and type(task_id) is int and task_id>=0,'invalid_libero_task')
        require(isinstance(language,str) and bool(language) and type(scene_revision) is int and scene_revision>0,'invalid_libero_context')
        new=(suite,task_id,language,scene_revision)
        if new!=self.context:self.invalidate()
        self.context=new
        return new

    def invalidate(self):
        self.discarded_actions+=len(self.queue);self.queue.clear();self.confirmed=None

    def set_mode(self,mode):
        require(mode in {'TASK_QUERY','CHANNEL_PROBE','CONFIRM'},'invalid_simulation_mode')
        self.mode=mode
        if mode!='CONFIRM':self.invalidate()

    def confirm(self,context,*,accepted,input_source):
        if accepted is not True:self.invalidate()
        require(input_source=='scripted_simulation_fixture','real_BCI_not_supported_by_simulation_gate')
        require(context==self.context and self.context is not None,'stale_simulation_confirmation')
        require(not self.stopped and self.mode=='CONFIRM' and accepted is True,'simulation_confirmation_not_accepted')
        self.confirmed=context

    def enqueue(self,actions,context):
        require(not self.stopped and self.mode=='CONFIRM' and context==self.context==self.confirmed and context is not None,'simulation_dispatch_not_authorized')
        require(isinstance(actions,list) and 0<len(actions)<=100,'invalid_simulation_chunk')
        for action in actions:
            require(isinstance(action,list) and len(action)==7,'libero_action_dimension_mismatch')
            require(all(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1 for v in action),'invalid_libero_normalized_action')
        self.queue.extend(tuple(action) for action in actions)

    def pop(self,context):
        require(not self.stopped and self.mode=='CONFIRM' and context==self.context==self.confirmed and context is not None,'simulation_action_context_invalid')
        require(bool(self.queue),'empty_simulation_queue')
        self.applied_actions+=1
        return list(self.queue.popleft())

    def stop(self):
        self.stopped=True;self.invalidate()

    def snapshot(self):
        return {'scope':'LIBERO_simulation_only','mode':self.mode,'context':self.context,'confirmed':self.confirmed,
            'stop_latched':self.stopped,'queued_actions':len(self.queue),'discarded_actions':self.discarded_actions,
            'applied_actions':self.applied_actions,'physical_execution_allowed':False,'physical_stop_verified':False}
