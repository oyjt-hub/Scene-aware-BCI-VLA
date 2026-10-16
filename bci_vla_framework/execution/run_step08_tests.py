"""Save exact joint-filter plus existing software checks, without collecting data."""
from pathlib import Path
import io
import json
import sys
import unittest

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(root/'tests'),top_level_dir=str(root))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(root/'execution/08_test_output.txt').write_bytes(b'\xef\xbb\xbf'+stream.getvalue().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
v={'date':'2026-10-16','scope':'offline_joint_filter_and_software_regression','command':'python -B execution/run_step08_tests.py','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'result':'PASS' if result.wasSuccessful() else 'FAIL','hardware_calls':0,'cloud_calls':0,'not_evaluated':['real EEG/calibration','real probe/task transfer','open-world model correctness','policy performance','G2-G4']}
(root/'execution/08_test_result.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,ensure_ascii=True,indent=2))
if not result.wasSuccessful():print(stream.getvalue())
raise SystemExit(0 if result.wasSuccessful() else 1)
