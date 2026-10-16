"""Save current software regressions without rewriting earlier test evidence."""
from pathlib import Path
import io
import json
import sys
import unittest

root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(root/'tests'),top_level_dir=str(root))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(root/'execution/11_test_output.txt').write_bytes(b'\xef\xbb\xbf'+stream.getvalue().replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
value={'date':'2026-10-16','scope':'offline_fake_lifecycle_and_software_regression',
    'command':'python -B execution/run_step11_tests.py','tests_run':result.testsRun,
    'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
    'result':'PASS' if result.wasSuccessful() else 'FAIL','hardware_calls':0,'cloud_calls':0,
    'not_evaluated':['real BCI or pi0.5','physical execution or stop','durable exactly-once delivery','step12 full episodes or G2-G4']}
(root/'execution/11_test_result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,ensure_ascii=True,indent=2))
if not result.wasSuccessful():
    print(stream.getvalue());sys.exit(1)
