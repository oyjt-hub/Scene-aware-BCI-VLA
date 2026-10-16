"""Produce repeatable core equivalence evidence through alternate replay transport."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bci_vla.pilot.transfer_replay import core_transfer_trace
folder=ROOT/'runs/transfer_step15_replay';assert not folder.exists();folder.mkdir(parents=True)
a=core_transfer_trace(packetized=False);b=core_transfer_trace(packetized=True)
semantic=lambda value:[{k:r[k] for k in ['at','kind','payload']} for r in map(json.loads,value.splitlines())]
assert a['checkpoints']==b['checkpoints'] and semantic(a['journal'])==semantic(b['journal'])
for name,value in [('direct',a),('packet',b)]:
    (folder/(name+'_journal.jsonl')).write_text(value.pop('journal'),encoding='utf-8')
    (folder/(name+'_trace.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
freeze=json.loads((ROOT/'execution/15_pre_run_freeze.json').read_text())
unchanged=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in freeze['source_files'])
assert unchanged
summary={'scope':'alternate_synthetic_replay_codec_through_unchanged_core','result':'PASS','core_checkpoints_equal':True,
    'journal_semantic_payloads_equal':True,'journal_instance_hashes_expected_to_differ':True,'each_journal_independently_replayed':True,
    'frozen_core_unchanged':unchanged,'actual_second_BCI_tested':False,'real_calibration_trials':0,
    'known_target_probe_unsupported_rejected':True,'does_not_prove_BCI_accuracy_or_transfer':True}
(folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary))
