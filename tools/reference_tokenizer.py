"""Load the reference tokenizer without silently replacing a requested slow class.

Transformers 5 prefers a remote fast class even with use_fast=False. An explicit
slow request therefore resolves the author's declared Python class directly.
Remote code still requires the caller's explicit trust; no tokenizer math changes.
"""


def load_tokenizer(model, revision=None, trust_remote_code=False, slow=False):
    from transformers import AutoTokenizer

    options = dict(revision=revision, trust_remote_code=trust_remote_code)
    if not slow:
        return AutoTokenizer.from_pretrained(model, **options)

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
