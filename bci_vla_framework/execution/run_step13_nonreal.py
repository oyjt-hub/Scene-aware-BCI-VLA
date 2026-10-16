"""Non-real-BCI checks only. All TCP servers use OS-assigned loopback ports."""
from pathlib import Path
import hashlib
import argparse
import json
import socket
import struct
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.software_flow import run_software_flow
from bci_vla.pilot.software_ui import luminance,frame_timing


def tcp_case(folder,kind):
    server=socket.socket();server.bind(('127.0.0.1',0));port=server.getsockname()[1]
    assert port!=8712
    payload=struct.pack('<'+str(12000)+'f',*(float(i%97) for i in range(12000))) if kind=='fragmented' else b''
    failures=[]
    def serve():
        try:
            connection,_=server.accept()
            with connection:
                if kind=='timeout':time.sleep(.45)
                else:
                    offset=0
                    while offset<len(payload):
                        size=[1,7,103,2048,4093][offset%5]
                        connection.sendall(payload[offset:offset+size]);offset+=size
        except Exception as exc:failures.append(repr(exc))
        finally:server.close()
    if kind=='refused':server.close();worker=None
    else:
        server.listen(1);server.settimeout(5)
        worker=threading.Thread(target=serve,daemon=True);worker.start()
    command=[sys.executable,'-B',str(ROOT/'execution/capture_step13_stream.py'),'--port',str(port),
        '--source-kind','synthetic_tcp_fixture','--seconds','.3','--output',str(folder)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=6)
    if worker:worker.join(6)
    assert not failures,failures
    report=json.loads((folder/'stream_report.json').read_text())
    assert report['recorder_connections_attempted']==0 and report['application_bytes_sent']==0
    if kind=='fragmented':
        assert result.returncode==0 and report['complete_frames']==1000 and report['partial_frame_tail_bytes']==0
        assert (folder/'raw_stream.bin').read_bytes()==payload
    else:
        assert result.returncode==2 and report['complete_frames']==0
        assert report['status']==('time_limit' if kind=='timeout' else 'connection_error')
    return {'case':kind,'passed':True,'returncode':result.returncode,'report_ref':str(folder.relative_to(ROOT)/'stream_report.json'),
        'frames':report['complete_frames'],'status':report['status']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/pilot_virtual_13_nonreal_acceptance_v3')
    folder=ROOT/parser.parse_args().output
    if folder.exists():raise SystemExit('Use existing evidence; this acceptance runner never overwrites it.')
    folder.mkdir(parents=True)
    flow,journal=run_software_flow()
    (folder/'session_journal.jsonl').write_text(journal,encoding='utf-8')
    (folder/'virtual_flow.json').write_text(json.dumps(flow,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tcp=[tcp_case(folder/('tcp_'+kind),kind) for kind in ['fragmented','timeout','refused']]
    schedules={str(f):[luminance(i,frequency_hz=f,refresh_hz=60) for i in range(240)] for f in [10,15]}
    timing={'ideal':frame_timing([i/60 for i in range(240)],refresh_hz=60),
        'injected_drop':frame_timing([0,1/60,2/60,4/60],refresh_hz=60)}
    assert timing['ideal']['long_intervals']==0 and timing['injected_drop']['long_intervals']==1
    summary={'scope':'non_real_BCI_software_only','result':'PASS','tcp_cases':tcp,'flow_final_state':flow['final_state'],
        'frame_timing':timing,'simulated_luminance':schedules,'frequencies_are_software_fixtures_only':True,
        'hardware_calls':0,'cloud_calls':0,'robot_commands':0,'real_pilot_trials':0,
        'actual_display_timing_measured':False,'decoder_accuracy':None,'G3_passed':False}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='simulated_luminance'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
