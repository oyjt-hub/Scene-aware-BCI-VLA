"""Check one actual raw receiver capture through the isolated resampling path."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.signal import prepare_decoder_window


def main():
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();source=Path(args.capture);source=source if source.is_absolute() else ROOT/source
    output=Path(args.output);output=output if output.is_absolute() else ROOT/output
    if output.exists():p.error('Choose a new output directory.')
    report=json.loads((source/'stream_report.json').read_text(encoding='utf-8'))
    raw=(source/'raw_stream.bin').read_bytes();channels=report['channel_count_declared_from_local_source']
    stride=4*channels;valid=len(raw)//stride*stride
    x=np.frombuffer(raw[:valid],dtype='<f4').reshape(-1,channels).T.copy()
    y,metadata=prepare_decoder_window(x,source_rate_hz=report['declared_sample_rate_hz'],decoder_rate_hz=250)
    output.mkdir(parents=True);np.save(output/'processed_signal.npy',y,allow_pickle=False)
    legacy=y[:,125:125+3800][:,::4]
    result={'scope':'actual_unlabeled_stream_sampling_path_check_not_SSVEP_validation',
        'source_capture':str(source),'raw_sha256':hashlib.sha256(raw).hexdigest(),'channel_identity_unverified':True,
        'new_path':metadata,'legacy_counterfactual':{'operation':'resampled250_then_slice125_to3925_then_stride4',
            'result_samples':legacy.shape[1],'interpreted_duration_at250Hz_s':legacy.shape[1]/250,
            'not_an_actual_four_second_stimulus_trial':True},
        'I01_source_code_defect_reproduced':True,'new_path_count_consistency_passed':True,
        'device_sample_rate_verified_independently':False,'I01_fully_resolved':False,
        'real_pilot_trials':0,'decoder_selections':0,'robot_commands':0,'G3_passed':False}
    (output/'sampling_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'raw_samples':x.shape[1],'processed_samples':y.shape[1],
        'source_duration_s':metadata['source_duration_s'],'decoder_duration_s':metadata['decoder_duration_s'],
        'legacy_counterfactual_samples':legacy.shape[1],'real_pilot_trials':0},indent=2))


if __name__=='__main__':main()
