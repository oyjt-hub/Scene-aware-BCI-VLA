"""OpenPI output to LIBERO controller input; no robot or network implementation."""
from ...protocols import require

def controller_actions(raw,*,lower,upper,replan_steps=8):
    import numpy as np
    x=np.asarray(raw,dtype=float);low=np.asarray(lower,dtype=float);high=np.asarray(upper,dtype=float)
    require(x.ndim==3 and x.shape[0]==1 and x.shape[2]==7 and x.shape[1]>=replan_steps,'pi05_action_shape_mismatch')
    require(type(replan_steps) is int and 0<replan_steps<=100,'invalid_replan_steps')
    require(low.shape==high.shape==(7,) and np.array_equal(low,-np.ones(7)) and np.array_equal(high,np.ones(7)),'unsupported_LIBERO_controller_bounds')
    require(np.isfinite(x).all(),'nonfinite_pi05_actions')
    selected=x[0,:replan_steps].copy();clipped=np.clip(selected,low,high)
    return clipped,{'scope':'LIBERO_controller_input_only','clipped_values':int(np.sum(selected!=clipped)),
        'raw_min':float(selected.min()),'raw_max':float(selected.max()),'lower':low.tolist(),'upper':high.tolist(),
        'rule':'explicit_controller_bound_clipping_no_extra_unnormalization_or_gripper_flip'}
