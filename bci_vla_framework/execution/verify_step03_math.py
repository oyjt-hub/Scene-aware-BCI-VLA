"""Exact finite identities for the design; no sampling, training or EEG experiment."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
examples = json.loads((root / "examples/alias_cases.json").read_text(encoding="utf-8"))
cases = {c["id"]: c for c in examples["cases"]}
checks = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def task(y, g, f, e, eps):
    return 1 - eps if y == (g ^ f ^ e) else eps

def probe(y, a, f, eps):
    return task(y, a, f, 0, eps)

def normalize(w):
    total = sum(w)
    if not total:
        raise ValueError("zero evidence probability; model mismatch")
    return [x / total for x in w]

states = [(0, 0), (0, 1), (1, 0), (1, 1)]
eps = F(1, 10)
check("example provenance excludes empirical interpretation", examples["provenance"] == "analytic_toy_examples_not_empirical_data")
check("single-step alias for all observations/encodings", all(task(y, 1, 0, e, ep) == task(y, 0, 1, e, ep) for y, e in product([0, 1], repeat=2) for ep in [F(0), eps, F(1, 2), F(1)]))
# Finite arithmetic sanity check; induction in the TXT is the general proof.
for n in range(1, 5):
    for ys in product([0, 1], repeat=n):
        for es in product([0, 1], repeat=n):
            a = b = F(1)
            for y, e in zip(ys, es):
                a *= task(y, 1, 0, e, eps)
                b *= task(y, 0, 1, e, eps)
            assert a == b
check("all fixed-encoding transcripts up to length 4 agree", True)
for n in range(1, 5):
    for ys in product([0, 1], repeat=n):
        a = b = F(1)
        for i, y in enumerate(ys):
            e = sum(ys[:i]) % 2
            ep = F(1, 10) if i % 2 else F(1, 5)
            a *= task(y, 1, 0, e, ep)
            b *= task(y, 0, 1, e, ep)
        assert a == b
check("history-dependent encoding and duration-dependent noise example", True)
check("known recoding cancels for every bit", all(((g ^ f ^ e ^ n) ^ e) == (g ^ f ^ n) for g, f, e, n in product([0, 1], repeat=4)))
check("double change can hide without conflict", task(0, 0, 0, 0, F(0)) == task(0, 1, 1, 0, F(0)) == 1)
for a in [0, 1]:
    assert probe(a, a, 0, F(0)) == 1 and probe(a, a, 1, F(0)) == 0
check("noiseless anchor separates the two modes", True)
c = cases["A04_anchor_noisy"]
check("noisy probe likelihood ratio", probe(0, 0, 0, eps) / probe(0, 0, 1, eps) == F(c["expected_likelihood_ratio_match"]))
posterior = normalize([probe(0, 0, 0, eps), probe(0, 0, 1, eps)])
check("one-match posterior", posterior[0] == F(c["expected_A_after_one_match"]))
check("two independent matches posterior", normalize([posterior[f] * probe(0, 0, f, eps) for f in [0, 1]])[0] == F(c["expected_A_after_two_independent_matches"]))
check("opposite evidence cancels", normalize([posterior[f] * probe(1, 0, f, eps) for f in [0, 1]])[0] == F(c["expected_A_after_match_then_mismatch"]))
check("nonuniform prior unchanged by task alias", normalize([F(4, 5) * task(1, 1, 0, 0, eps), F(1, 5) * task(1, 0, 1, 0, eps)])[0] == F(4, 5))
check("random probe is uninformative", all(probe(y, a, 0, F(1, 2)) == probe(y, a, 1, F(1, 2)) for y, a in product([0, 1], repeat=2)))
for eta in [F(0), F(1, 5), F(1, 2), F(1)]:
    observed_error = eta * (1 - eps) + (1 - eta) * eps
    assert observed_error == eta + eps - 2 * eta * eps
check("compliance composite error formula", True)
check("unknown persistent compliance flip aliases channel flip", all((a ^ 0 ^ 1) == (a ^ 1 ^ 0) for a in [0, 1]))
case = cases["A05_joint_correlation"]
b = normalize([F(1, 4) * task(1, g, f, 0, eps) for g, f in states])
check("four-state task posterior", b == [F(x) for x in case["expected_after_task"]])
b2 = normalize([w * probe(0, 0, f, eps) for w, (g, f) in zip(b, states)])
check("four-state probe posterior", b2 == [F(x) for x in case["expected_after_probe"]])
check("probe indirectly updates goal marginal", sum(w for w, (g, f) in zip(b2, states) if g == 1) == F(case["expected_goal1_marginal"]) != F(1, 2))
alias_joint = [F(0), F(1, 2), F(1, 2), F(0)]
goal_marginal = [sum(w for w, (g, f) in zip(alias_joint, states) if g == k) for k in [0, 1]]
channel_marginal = [sum(w for w, (g, f) in zip(alias_joint, states) if f == k) for k in [0, 1]]
factored = [goal_marginal[g] * channel_marginal[f] for g, f in states]
check("factorization invents impossible support", alias_joint[0] == alias_joint[3] == 0 and factored[0] == factored[3] == F(1, 4) and sum(factored) == 1)
check("positive likelihood cannot revive zero prior", normalize([F(0), F(1)])[0] == 0)
try:
    normalize([F(0), F(0)])
except ValueError:
    check("zero normalizer is an error", True)
else:
    raise AssertionError("must detect zero normalizer")
result = {"scope": "analytic_identity_check", "result": "PASS", "count": len(checks), "checks": checks, "general_proof": "docs/03_identifiability.txt sections 2-5, conditional on stated assumptions", "not_evaluated": ["EEG", "probe usability", "task/probe transfer", "novelty", "policy performance", "G2-G4"]}
text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
(root / "execution/03_math_result.json").write_text(text, encoding="utf-8")
print(text)
