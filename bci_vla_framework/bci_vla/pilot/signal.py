"""Versioned single-resampling path; raw units/channel identity remain external."""
from fractions import Fraction
from ..protocols import require,number


def prepare_decoder_window(samples,*,source_rate_hz,decoder_rate_hz=250,start_s=0.,duration_s=None):
    import numpy as np
    from scipy.signal import butter,sosfiltfilt,iirnotch,filtfilt,resample_poly
    x=np.asarray(samples,dtype=float)
    require(x.ndim==2 and x.shape[0]>0 and x.shape[1]>32,'channel_by_sample_array_required')
    require(np.isfinite(x).all(),'nonfinite_source_signal')
    number(source_rate_hz,1);number(decoder_rate_hz,1);number(start_s,0)
    require(source_rate_hz>200 and decoder_rate_hz>200,'filter_requires_nyquist_above100Hz')
    start=int(round(start_s*source_rate_hz))
    if duration_s is None:end=x.shape[1]
    else:
        number(duration_s,0);require(duration_s>0,'positive_window_duration_required')
        end=start+int(round(duration_s*source_rate_hz))
    require(0<=start<end<=x.shape[1] and end-start>32,'requested_window_outside_source')
    # Slice in the SOURCE timebase, then filter and resample exactly once.
    window=x[:,start:end]
    band=butter(4,[.3,90],btype='bandpass',fs=source_rate_hz,output='sos')
    filtered=sosfiltfilt(band,window,axis=1)
    b,a=iirnotch(50,30,fs=source_rate_hz);filtered=filtfilt(b,a,filtered,axis=1)
    ratio=Fraction(str(decoder_rate_hz))/Fraction(str(source_rate_hz))
    require(ratio.numerator<=10000 and ratio.denominator<=10000,'unsupported_resampling_ratio')
    result=resample_poly(filtered,ratio.numerator,ratio.denominator,axis=1)
    expected=(end-start)*decoder_rate_hz/source_rate_hz
    require(abs(result.shape[1]-expected)<=1,'resample_length_mismatch')
    return result,{'version':'pilot-single-resample/1.0','source_rate_hz':source_rate_hz,'decoder_rate_hz':decoder_rate_hz,
        'source_start_sample':start,'source_end_sample':end,'source_samples':end-start,'decoder_samples':result.shape[1],
        'source_duration_s':(end-start)/source_rate_hz,'decoder_duration_s':result.shape[1]/decoder_rate_hz,
        'bandpass_hz':[.3,90],'notch_hz':50,'anti_alias':'scipy.signal.resample_poly default FIR',
        'resample_operations':1,'legacy_250_to62_5_stride_removed':True,'units_verified':False,
        'online_filter_latency_validated':False,'real_decoder_accuracy_validated':False}
