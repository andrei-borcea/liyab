"""Linear-mixer facts must explain a whole observed call; run with a PyTorch Python."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reference_trace import Call, analyze_block, analyze_linear_mixer


class DirectForward(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.projection = nn.Linear(7, 7, bias=bias)

    def forward(self, x):
        return self.projection.forward(x)


class LinearMixerFacts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(317)
        self.x = torch.randn(2, 13, 7)

    def fact(self, module, output=None):
        return analyze_linear_mixer(Call(1, "block.mixer", module, [self.x], {},
                                         [module(self.x) if output is None else output]))

    def test_direct_linear(self):
        self.assertEqual(self.fact(nn.Linear(7, 7))["projection"], "block.mixer")

    def test_child_forward_bypassing_hooks(self):
        module = DirectForward()
        observed = []
        handle = module.projection.register_forward_hook(lambda *args: observed.append(True))
        try:
            self.assertEqual(self.fact(module)["projection"], "block.mixer.projection")
        finally:
            handle.remove()
        self.assertEqual(observed, [])

    def test_a_linear_only_residual_branch_keeps_its_norm_and_absent_ffn(self):
        module = DirectForward()
        norm = nn.RMSNorm(7, eps=1e-5)
        normalized = norm(self.x)
        mixed = module(normalized)
        calls = [Call(1, "block.norm", norm, [self.x], {}, [normalized]),
                 Call(2, "block.mixer", module, [normalized], {}, [mixed])]
        layer = Call(0, "block", nn.Module(), [self.x], {}, [self.x + mixed])
        fact = analyze_block(SimpleNamespace(attn=[]), 0, "block", layer, calls, [], 1e-5)
        self.assertEqual(fact["mixer"]["kind"], "linear")
        self.assertEqual(fact["ffn"]["kind"], "none")
        self.assertIsNotNone(fact["norm_before_mixer"])
        self.assertEqual(fact["unknown"], [])

    def test_bias_is_recomputed(self):
        self.assertEqual(self.fact(DirectForward(bias=True))["kind"], "linear")

    def test_extra_nonlinearity_is_unknown(self):
        module = DirectForward()
        self.assertIsNone(self.fact(module, torch.nn.functional.silu(module(self.x))))

    def test_extra_residual_is_unknown(self):
        module = DirectForward()
        self.assertIsNone(self.fact(module, module(self.x) + self.x))

    def test_duplicate_candidate_is_unknown_even_when_unused(self):
        module = DirectForward()
        module.other = nn.Linear(7, 7, bias=False)
        module.other.load_state_dict(module.projection.state_dict())
        self.assertIsNone(self.fact(module))

    def test_shape_change_is_unknown(self):
        module = DirectForward()
        self.assertIsNone(self.fact(module, module(self.x).transpose(0, 1)))

    def test_incompatible_projection_is_unknown(self):
        module = nn.Linear(5, 7)
        self.assertIsNone(self.fact(module, self.x))


if __name__ == "__main__":
    unittest.main()
