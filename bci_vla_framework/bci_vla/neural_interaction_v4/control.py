from bci_vla.interaction_contract_v4.validation import control_route

def route_control(public):
 """Pure contract routing; actual existing STOP/queue cancellation is external."""
 history=public.get('history',[])
 if history and history[-1]['role'] in ['STOP','CANCEL']:
  return {'route':control_route(history[-1]),'call_neural_module':False,'execution_authorized':False}
 if public.get('public_state',{}).get('stop_latched'):
  return {'route':'STOP_DIRECT_EXISTING_PATH','call_neural_module':False,'execution_authorized':False}
 return {'route':'NEURAL_PROPOSAL_ONLY','call_neural_module':True,'execution_authorized':False}
