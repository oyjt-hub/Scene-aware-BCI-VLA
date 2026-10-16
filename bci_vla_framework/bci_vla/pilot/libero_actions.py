"""StarVLA LIBERO conversion, matching the checked repository evaluation path."""
from ..protocols import require

def decode_starvla_chunk(raw,stats):
    import numpy as np
    x=np.asarray(raw,dtype=float)
    require(x.ndim==3 and x.shape[0]==1 and x.shape[2]==7 and 0<x.shape[1]<=100,'invalid_starvla_chunk_shape')
    require(np.isfinite(x).all(),'nonfinite_starvla_chunk')
    low=np.asarray(stats['min'],dtype=float);high=np.asarray(stats['max'],dtype=float);mask=np.asarray(stats['mask'])
    require(low.shape==high.shape==mask.shape==(7,) and np.isfinite(low).all() and np.isfinite(high).all(),'invalid_libero_action_statistics')
    require(mask.dtype==np.bool_ and not bool(mask[6]) and np.all(low<high),'unsupported_libero_action_statistics')
    normalized=np.clip(x[0],-1,1)
    normalized[:,6]=np.where(normalized[:,6]<.5,0,1)
    actions=np.where(mask,.5*(normalized+1)*(high-low)+low,normalized)
    # StarVLA encodes open=1; LIBERO's action convention requires open=-1.
    actions[:,6]=1-2*(actions[:,6]>.5)
    return actions
