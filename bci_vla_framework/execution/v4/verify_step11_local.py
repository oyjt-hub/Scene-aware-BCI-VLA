"""Persist reproducible local mechanism checks; never substitute for real VLM."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT.parent/'.runtime/step06_cpu'),str(ROOT)]
OUT=ROOT/'execution/v4'
suite=unittest.TestSuite()
for pattern in ('test_step11_authorization_v4.py','test_upstream_subtask_v4.py'):
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
log=stream.getvalue();print(log)
(OUT/'11_local_tests.log').write_text(log,encoding='utf-8')
report={'tests':result.testsRun,'successful':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
        'log_sha256':hashlib.sha256((OUT/'11_local_tests.log').read_bytes()).hexdigest(),
        'scope':'local authorization and exact upstream method mechanics using explicit scripted tiny-model stubs',
        'actual_VLM_inference':False,'upstream_trained_checkpoint':False,'formal_test_accessed':False,'action_sampling':False}
(OUT/'11_local_tests.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(not result.wasSuccessful())
