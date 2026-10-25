"""Checkpoint inspection agrees across the two Hugging Face storage formats.

Run with the reference environment: python tests/test_describe_model.py.
The engine's standard-library-only registry self-test does not require torch.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from describe_model import Checkpoint, FormatError  # noqa: E402


class CheckpointTest(unittest.TestCase):
    def test_pytorch_and_safetensors_have_identical_rows(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as files:
            safe, binary = Path(tmp) / "safe", Path(tmp) / "binary"
            safe.mkdir()
            binary.mkdir()
            state = {
                "model.norm.weight": torch.tensor([1.0, -0.0, 0.25]),
                "model.layers.0.k_proj.weight": torch.arange(35).reshape(7, 5).to(torch.bfloat16) / 8,
                "model.layers.0.scalar": torch.tensor(0.125, dtype=torch.float16),
            }
            save_file(state, safe / "model.safetensors")
            torch.save(state, binary / "pytorch_model.bin")
            a, b = Checkpoint(str(safe)), Checkpoint(str(binary))
            for source, _, _ in a.files.values():
                files.callback(source.file.close)
            self.assertEqual(a.find("norm.weight"), b.find("norm.weight"))
            for name in state:
                rows = 1 if state[name].ndim == 0 else state[name].shape[0]
                for first, count in ((0, None), (0, 1), (rows, 0), (0, 0), (rows - 1, 1)):
                    x, y = a.tensor(name, first, count), b.tensor(name, first, count)
                    self.assertEqual(x.shape, y.shape)
                    np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))
                for reader in (a, b):
                    for first, count in ((-1, 1), (0, -1), (rows, 1)):
                        with self.assertRaises(FormatError):
                            reader.tensor(name, first, count)

    def test_pytorch_shards_are_loaded_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"layer.weight": torch.tensor([0.5, 1.5])}, root / "part-1.bin")
            (root / "part-2.bin").write_bytes(b"unused shard is deliberately unreadable")
            (root / "pytorch_model.bin.index.json").write_text(json.dumps({"weight_map": {
                "layer.weight": "part-1.bin", "unused.weight": "part-2.bin",
            }}))
            reader = Checkpoint(tmp)
            np.testing.assert_array_equal(reader.tensor("layer.weight"), [0.5, 1.5])

    def test_missing_or_non_tensor_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FormatError, "no safetensors or PyTorch checkpoint"):
                Checkpoint(tmp)
            torch.save({"not_a_weight": "text"}, Path(tmp) / "pytorch_model.bin")
            with self.assertRaisesRegex(FormatError, "tensor state dictionary"):
                Checkpoint(tmp)


if __name__ == "__main__":
    unittest.main()
