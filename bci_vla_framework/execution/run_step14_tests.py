"""Step14 simulator bridge regression; preserve old test reports."""
from pathlib import Path
import io,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(ROOT/'execution/14_test_output.txt').write_bytes(b'\xef\xbb\xbf'+stream.getvalue().replace('\r\n','\n').replace('\n','\r\n').encode())
v={'date':'2026-10-16','scope':'step14_LIBERO_gate_and_frozen_regressions','tests_run':result.testsRun,
    'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'result':'PASS' if result.wasSuccessful() else 'FAIL',
    'physical_robot_commands':0,'real_BCI_trials':0,'does_not_prove_G3_or_full_framework_integration':True}
(ROOT/'execution/14_test_result.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if not result.wasSuccessful():print(stream.getvalue());sys.exit(1)
