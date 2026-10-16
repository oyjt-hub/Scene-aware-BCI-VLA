"""Run protocol software tests and save the full output; no benchmark data."""
from pathlib import Path
import io
import json
import sys
import time
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
stream = io.StringIO()
start = time.monotonic()
suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
(root / "execution/05_test_output.txt").write_bytes(b"\xef\xbb\xbf" + stream.getvalue().replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
record = {"scope": "software_protocol_replay", "command": "python -B execution/run_step05_tests.py", "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "result": "PASS" if result.wasSuccessful() else "FAIL", "duration_s": time.monotonic() - start, "hardware_calls": 0, "cloud_calls": 0, "evidence": "execution/05_test_output.txt", "not_evaluated": ["EEG validity", "live transport", "clock calibration", "channel likelihoods", "belief filtering", "robot execution", "G2-G4"]}
(root / "execution/05_test_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
if not result.wasSuccessful(): print(stream.getvalue())
raise SystemExit(0 if result.wasSuccessful() else 1)
