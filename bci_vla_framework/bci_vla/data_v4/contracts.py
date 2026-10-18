"""4.1 data validator using the unchanged 4.0 cross-field semantic checks."""
from pathlib import Path
from bci_vla.interaction_contract_v4.validation import _shape,_public,_private,load_json
SCHEMA=Path(__file__).resolve().parents[2]/'configs/v4/interfaces_data_4_1.json'
def validate_public(value):
 defs=load_json(SCHEMA)['$defs'];_shape(value,defs['public_input'],defs);_public(value);return True
def validate_private(value,public):
 defs=load_json(SCHEMA)['$defs'];_shape(value,defs['private_labels'],defs);_private(value,public);return True
def iter_public(root,partition):
 if partition not in ['train','validation','calibration_fit','calibration_audit']:raise ValueError('Formal test is sealed')
 path=Path(root)/partition/'public_input.jsonl'
 import json
 for line in path.open(encoding='utf-8'):
  value=json.loads(line);validate_public(value);yield value
