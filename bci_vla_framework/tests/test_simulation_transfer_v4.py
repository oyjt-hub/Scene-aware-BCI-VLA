"""Migration semantics without substituting toy evidence for model tests."""
import unittest
from copy import deepcopy
from unittest.mock import patch
from tests.test_simulation_v3 import fixtures
from bci_vla.simulation_transfer_v4 import factory
from bci_vla.simulation_transfer_v4.interaction import resolve,dispatch,endpoint
from bci_vla.simulation_v3.replay import replay_session_state
from bci_vla.benchmark.offline import stamp

class MigrationTests(unittest.TestCase):
 def execute(self,transport,executor):
  e,b,scene,out=fixtures();out=deepcopy(out);out['executor_id']=executor
  with patch.object(factory,'TRANSPORT',transport):s,c=factory.make_runtime('unit',scene,b,out,prior=(.9,0,.1,0))
  # Codec equivalence holds for identical clocks, not independently timed runs.
  with patch('bci_vla.benchmark.offline.perf_counter_ns',return_value=0):resolve(c,0,'normal')
  dispatch(s,c,e,'unit-command');b.advance();c.now+=.1;s.refresh_submission(now=stamp(c.now))
  b.finish();c.now+=.1;s.refresh_submission(now=stamp(c.now));endpoint(s,c,e,'unit')
  self.assertEqual(s.state,'SUCCEEDED');self.assertEqual(replay_session_state(s.journal),s.snapshot())
  return s,c

 def test_direct_packet_preserve_every_core_journal_payload(self):
  a,c=self.execute('direct','Pi05-LIBERO-v4');b,d=self.execute('packet','Pi05-LIBERO-v4')
  self.assertEqual(a.snapshot(),b.snapshot())
  self.assertEqual([x['payload'] for x in a.journal.records],[x['payload'] for x in b.journal.records])
  self.assertEqual(getattr(c,'packet_events',0),0);self.assertGreater(d.packet_events,0)

 def test_executor_binding_comes_from_independent_outcomes(self):
  for executor in ['Pi05-LIBERO-v4','StarVLA-LIBERO-v4']:
   s,c=self.execute('packet',executor)
   self.assertEqual(s.config['backend_id'],executor)

 def test_holdout_changes_do_not_change_fitted_outcome_rows(self):
  rows=[dict(split='calibration',record_id='cal'+str(i),target_index=i,held={o:j==i for j,(o,_) in enumerate(factory.OBJECTS)},simulator_reset_wall_s=.1) for i in range(2)]
  a=factory.empirical_outcomes(rows,'c'*64)
  rows.append(dict(split='calibration_holdout',target_index=0,held={o:False for o,_ in factory.OBJECTS}))
  self.assertEqual(a,factory.empirical_outcomes(rows,'c'*64))
