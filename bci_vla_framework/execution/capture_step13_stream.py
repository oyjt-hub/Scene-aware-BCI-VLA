"""One bounded receive-only local recorder check. No stimulus, decoding or action."""
from pathlib import Path
import argparse
import hashlib
import json
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.neuracle_stream import FloatFrameParser


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8712);p.add_argument('--channels',type=int,default=12)
    p.add_argument('--declared-rate',type=float,default=1000);p.add_argument('--seconds',type=float,default=3)
    p.add_argument('--source-kind',choices=['live_recorder_stream','synthetic_tcp_fixture'],default='live_recorder_stream')
    p.add_argument('--output',required=True);args=p.parse_args()
    if args.source_kind=='synthetic_tcp_fixture' and args.port==8712:p.error('Synthetic checks must never use the recorder port.')
    if not 0<args.seconds<=10:p.error('Duration must be bounded to10 seconds.')
    if args.declared_rate<=0:p.error('Declared rate must be positive.')
    folder=Path(args.output)
    if not folder.is_absolute():folder=ROOT/folder
    if folder.exists():p.error('Use a new output directory; existing evidence is never overwritten.')
    folder.mkdir(parents=True);parser=FloatFrameParser(args.channels)
    packets=[];raw=bytearray();first=None;last=None;status='time_limit';error=None
    start=time.perf_counter();deadline=start+args.seconds
    try:
        with socket.create_connection(('127.0.0.1',args.port),timeout=1) as stream:
            stream.settimeout(.25)
            while time.perf_counter()<deadline and len(raw)<1_000_000:
                stream.settimeout(max(.001,min(.25,deadline-time.perf_counter())))
                try:data=stream.recv(min(65536,1_000_000-len(raw)))
                except socket.timeout:continue
                if not data:status='remote_closed';break
                now=time.perf_counter();first=now if first is None else first;last=now
                raw.extend(data);frames=parser.feed(data)
                packets.append({'receive_elapsed_s':now-start,'bytes':len(data),'parsed_frames':len(frames)})
    except OSError as exc:status='connection_error';error=type(exc).__name__+':'+str(exc)
    elapsed=time.perf_counter()-start
    (folder/'raw_stream.bin').write_bytes(raw)
    period=None if len(packets)<2 else last-first
    first_frames=packets[0]['parsed_frames'] if packets else 0
    report={'scope':('synthetic_stream_transport_check' if args.source_kind=='synthetic_tcp_fixture' else 'live_unlabeled_receive_only_stream_check_not_pilot_trial'),'source_kind':args.source_kind,'host':'127.0.0.1','port':args.port,
        'status':status,'error':error,'channel_count_declared_from_local_source':args.channels,
        'declared_sample_rate_hz':args.declared_rate,'decoded_format':'little_endian_float32_interleaved_assumed_from_local_source',
        'bytes_received':len(raw),'complete_frames':parser.frames,'partial_frame_tail_bytes':len(parser.tail),
        'nonfinite_values':parser.nonfinite,'elapsed_s':elapsed,'packets':packets,
        'throughput_rate_estimate_hz':(parser.frames-first_frames)/period if period and period>0 else None,
        'throughput_is_not_hardware_clock_verification':True,'channel_names_verified':False,'units_verified':False,
        'raw_sha256':hashlib.sha256(raw).hexdigest(),'application_bytes_sent':0,
        'recorder_connections_attempted':int(args.source_kind=='live_recorder_stream'),'synthetic_loopback_connections':int(args.source_kind=='synthetic_tcp_fixture'),'stimulus_presented':False,'decoded_selection':None,
        'pilot_trials':0,'calibration_trials':0,'robot_commands':0,'server_or_cloud_calls':0,'G3_passed':False}
    (folder/'stream_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['scope','status','bytes_received','complete_frames','nonfinite_values',
        'throughput_rate_estimate_hz','application_bytes_sent','pilot_trials']},indent=2))
    if not raw:sys.exit(2)


if __name__=='__main__':main()
