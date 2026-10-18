"""Local tests of optional language-remediation LoRA, not trained capability."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from bci_vla.high_level_v4.lora import HighLevelLoRALinear,install_high_level_lora,high_level_scope,adapter_state

class LoRATests(unittest.TestCase):
 def test_exact_off_and_initial_zero_update(self):
  torch.manual_seed(4);base=nn.Linear(12,8);x=torch.randn(2,12);expected=base(x).detach().clone();module=HighLevelLoRALinear(base,rank=4,alpha=8)
  self.assertTrue(torch.equal(module(x),expected));module.enabled=True;self.assertTrue(torch.equal(module(x),expected))
  module(x).square().mean().backward();self.assertGreater(float(module.B.grad.abs().sum()),0.);self.assertIsNone(base.weight.grad)
  with torch.no_grad():module.B.add_(.1)
  self.assertGreater(float((module(x)-expected).abs().max()),0.);module.enabled=False;self.assertTrue(torch.equal(module(x),expected))
 def test_scope_restored_on_failure_and_only_language_targets(self):
  class Layer(nn.Module):
   def __init__(self):
    super().__init__();self.self_attn=nn.Module()
    for name in ['q_proj','v_proj','o_proj']:setattr(self.self_attn,name,nn.Linear(8,8))
  language=nn.Module();language.layers=nn.ModuleList([Layer(),Layer()]);expert=nn.Linear(8,8)
  model=SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(language_model=language),gemma_expert=expert))
  old=expert.weight.detach().clone();adapters=install_high_level_lora(model,rank=2)
  self.assertEqual(len(adapters),6)
  with self.assertRaises(RuntimeError):
   with high_level_scope(adapters):
    self.assertTrue(all(v.enabled for v in adapters.values()));raise RuntimeError('test')
  self.assertTrue(all(not v.enabled for v in adapters.values()));self.assertTrue(torch.equal(old,expert.weight));self.assertEqual(len(adapter_state(adapters)),12)
  with self.assertRaises(ValueError):install_high_level_lora(model,rank=2)
if __name__=='__main__':unittest.main(verbosity=2)
