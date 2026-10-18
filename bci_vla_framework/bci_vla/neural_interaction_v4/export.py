"""Prototype inspection format, NOT the executable frozen V4-02 action contract."""
from .losses import ACTIONS

def inspect_prediction(output,metadata,index=0):
 valid=bool(output['reliability_valid'][index])
 probabilities=output['candidate_probabilities'][index].detach().cpu().tolist()
 return {'schema_version':'bci-vla-module-inspection/0.1',
  'sample_id':metadata[index]['sample_id'],'input_sha256':metadata[index]['input_sha256'],
  'prefix_shape':list(output['prefix'][index].shape),
  'candidates':[{'instance_id':oid,'raw_probability':probabilities[j]} for j,oid in enumerate(metadata[index]['instance_ids'])],
  'outside_raw_probability':float(output['outside_probability'][index].detach()),
  'channel_raw_probability':float(output['reliability_probability'][index].detach()) if valid else None,
  'channel_status':'SUPPORTED_UNCALIBRATED' if valid else 'INSUFFICIENT_OR_EXPIRED',
  'interaction_raw_action':ACTIONS[int(output['interaction_logits'][index].argmax())],
  'model_trained':False,'probability_calibrated':False,'execution_authorized':False,
  'scope':'Random-weight CPU interface inspection only; no language generation or VLA dispatch'}
