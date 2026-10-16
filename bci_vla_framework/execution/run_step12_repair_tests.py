"""Versioned authorization repair regression without overwriting step12 results."""
from pathlib import Path
import io
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(ROOT/'execution/12R_test_output.txt').write_bytes(b'\xef\xbb\xbf'+stream.getvalue().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
v={'date':'2026-10-16','scope':'authorization_aware_runtime_v2_and_all_frozen_regressions',
    'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
    'result':'PASS' if result.wasSuccessful() else 'FAIL','hardware_calls':0,'cloud_calls':0,
    'software_pass_does_not_grant_G2':True}
(ROOT/'execution/12R_test_result.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,indent=2))
if not result.wasSuccessful():print(stream.getvalue());sys.exit(1)
