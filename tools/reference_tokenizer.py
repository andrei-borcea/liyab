"""Load the reference tokenizer without silently replacing a requested slow class.

Transformers 5 prefers a remote fast class even with use_fast=False. An explicit
slow request therefore resolves the author's declared Python class directly.
Remote code still requires the caller's explicit trust. Standard fast classes
retain the author's serialized normalization and splitting if transformers
reconstructs a different backend from vocab/merges.
"""


def load_tokenizer(model, revision=None, trust_remote_code=False, slow=False):
    from transformers import AutoTokenizer

    options = dict(revision=revision, trust_remote_code=trust_remote_code)
    if not slow:
        tokenizer = AutoTokenizer.from_pretrained(model, **options)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            return tokenizer
        # Some transformers 5 classes reconstruct vocab/merges and lose the
        # serialized normalizer or pre-tokenizer. The tokenizer JSON is also
        # the independent three-text gate's source of truth. Reuse the public
        # tokenizer_object API to preserve that artifact, retaining the same
        # class, special-token configuration and chat template.
        import json
        import os

        path = os.path.join(model, "tokenizer.json")
        if not os.path.isfile(path):
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model, "tokenizer.json", revision=revision)
            except Exception:  # no JSON: the declared tokenizer class owns its construction
                return tokenizer
        from tokenizers import Tokenizer

        # Parse both pipelines through the same runtime: older JSON omits
        # default fields (for example ByteLevel.use_regex=true). Comparing raw
        # JSON would mistake those equivalent defaults for changed splitting.
        source_backend = Tokenizer.from_file(path)
        serialized = json.loads(source_backend.to_str())
        fields = ("normalizer", "pre_tokenizer")
        actual = json.loads(backend.to_str())
        if all(actual.get(field) == serialized.get(field) for field in fields):
            return tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, tokenizer_object=source_backend, **options)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        actual = json.loads(backend.to_str()) if backend is not None else {}
        if backend is None or any(actual.get(field) != serialized.get(field) for field in fields):
            raise ValueError("the reference tokenizer class did not retain its source JSON normalization and splitting")
        return tokenizer

    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    config = get_tokenizer_config(model, revision=revision)
    mapping = config.get("auto_map", {})
    mapping = mapping.get("AutoTokenizer") if isinstance(mapping, dict) else mapping
    if mapping and mapping[0] and trust_remote_code:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(mapping[0], model, revision=revision)
        tokenizer = cls.from_pretrained(model, **options)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, **options)
    if getattr(tokenizer, "is_fast", False):
        raise ValueError("the requested original slow tokenizer is unavailable; the loader selected a fast class")
    return tokenizer
