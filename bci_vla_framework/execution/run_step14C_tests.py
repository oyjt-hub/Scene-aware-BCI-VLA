"""Preserve prior reports while testing new transfer preparation."""
from pathlib import Path
import io,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT))
r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(ROOT/'execution/14C_test_output.txt').write_bytes(b'\xef\xbb\xbf'+stream.getvalue().replace('\r\n','\n').replace('\n','\r\n').encode())
v={'date':'2026-10-16','scope':'step14C_full_simulator_lifecycle_and_frozen_regressions','tests_run':r.testsRun,'failures':len(r.failures),
    'errors':len(r.errors),'skipped':len(r.skipped),'result':'PASS' if r.wasSuccessful() else 'FAIL','real_BCI_trials':0,'physical_robot_commands':0}
(ROOT/'execution/14C_test_result.json').write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if not r.wasSuccessful():print(stream.getvalue());sys.exit(1)
