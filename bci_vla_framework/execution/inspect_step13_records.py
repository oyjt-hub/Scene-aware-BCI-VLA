"""Inspect externally collected private calibration/pilot JSONL; never acquire."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.validation import summarize_records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--software-fixture',action='store_true',help='Explicitly permit synthetic software fixtures; cannot count as real pilot.')
    args=parser.parse_args();source=Path(args.records);destination=Path(args.output)
    if destination.exists():parser.error('Output exists; choose a new output path to preserve evidence.')
    data=source.read_bytes();rows=[json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]
    result=summarize_records(rows,allow_synthetic=args.software_fixture)
    result.update(source_sha256=hashlib.sha256(data).hexdigest(),source_path=str(source.resolve()),
        hardware_calls=0,cloud_calls=0,physical_execution_allowed=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'scope':result['scope'],'records':result['records'],'real_records':result['real_records'],
        'model_fitted':False,'G3_passed':False,'output':str(destination.resolve())},indent=2))


if __name__=='__main__':main()
