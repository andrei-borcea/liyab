"""Reference loaders must preserve serialized normalization without replacing slow tokenizers.

Run with python3 tests/test_reference_tokenizer.py; no model weights or third-party packages are needed.
"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reference_tokenizer import load_tokenizer


class SerializedPipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.source = {"normalizer": {"type": "NFKC"}, "pre_tokenizer": {"type": "ByteLevel"}}
        (self.path / "tokenizer.json").write_text(json.dumps(self.source))
        self.actual = self.source.copy()
        self.calls = []
        self.retain = True

        class Backend:
            def __init__(self, data): self.data = data
            def to_str(self): return json.dumps(self.data)

        def construct(model, **options):
            self.calls.append((model, options))
            backend = options.get("tokenizer_object") if self.retain else None
            return types.SimpleNamespace(backend_tokenizer=backend or Backend(self.actual))

        self.backend = Backend
        modules = {
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=construct)),
            "tokenizers": types.SimpleNamespace(Tokenizer=types.SimpleNamespace(
                from_file=lambda path: Backend(json.loads(Path(path).read_text())))),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_matching_pipeline_keeps_original_construction(self):
        load_tokenizer(str(self.path), revision="pinned", trust_remote_code=False)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], {"revision": "pinned", "trust_remote_code": False})

    def test_lost_normalizer_reloads_the_source_object(self):
        self.actual = {**self.source, "normalizer": None}
        result = load_tokenizer(str(self.path), revision="pinned", trust_remote_code=True)
        self.assertEqual(json.loads(result.backend_tokenizer.to_str()), self.source)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1][1]["revision"], "pinned")
        self.assertTrue(self.calls[1][1]["trust_remote_code"])

    def test_lost_splitting_reloads_the_source_object(self):
        self.actual = {**self.source, "pre_tokenizer": None}
        result = load_tokenizer(str(self.path))
        self.assertEqual(json.loads(result.backend_tokenizer.to_str()), self.source)
        self.assertEqual(len(self.calls), 2)

    def test_a_class_discarding_the_source_is_refused(self):
        self.actual = {**self.source, "normalizer": None}
        self.retain = False
        with self.assertRaisesRegex(ValueError, "did not retain"):
            load_tokenizer(str(self.path))

    def test_an_unavailable_json_keeps_the_declared_class(self):
        (self.path / "tokenizer.json").unlink()
        with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(
                hf_hub_download=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")))}):
            load_tokenizer(str(self.path))
        self.assertEqual(len(self.calls), 1)

    def test_a_reference_without_a_fast_backend_is_left_to_its_class(self):
        def construct(*args, **kwargs):
            self.calls.append(kwargs)
            return types.SimpleNamespace()
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=construct))}):
            result = load_tokenizer(str(self.path))
        self.assertFalse(hasattr(result, "backend_tokenizer"))
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
