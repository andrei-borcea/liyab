#!/usr/bin/env python3
"""Writes reference embeddings and reranker scores from Hugging Face transformers, for liyab-embed.

Each model is run the way its model card runs it, in float32:

  * an embedding model (--embedding, e.g. Qwen/Qwen3-Embedding-0.6B): its
    tokenizer (whose post-processor appends <|endoftext|>), the base model's
    last hidden state (after the final norm), the last token's row, L2
    normalization. Also written: the mean of every token's row (normalized),
    for Liyab's Mean pooling, and the last token's row as is (not normalized);
  * a reranker (--reranker, e.g. Qwen/Qwen3-Reranker-0.6B): the card's
    system prefix, "<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>:
    {document}" and the assistant suffix, the causal LM's logits of "yes" and
    "no" at the last position, and P(yes) = softmax(no, yes)[yes], the card's
    normalization.

Inputs come from a cases file (tests/data/pooling_cases.json): "embed", texts
embedded as they are (queries carry their instruction), and "rerank", cases of
one query with its documents. Each input runs alone, without padding, so the
reference holds no padding effects; --check-batched also runs the card's
left-padded batches and prints how far they are from it (float32 rounding).
--sentence-transformers also runs sentence-transformers, when installed, and
prints how far it is from the transformers path: the embedding model's
SentenceTransformer (its pooling and normalization modules) and the
reranker's CrossEncoder (whose score is the "yes" minus "no" logit: its
sigmoid is the card's P(yes)).

    tools/pooling_reference.py --embedding Qwen/Qwen3-Embedding-0.6B --reranker Qwen/Qwen3-Reranker-0.6B \\
        --cases tests/data/pooling_cases.json --out pooling.ref.json
    liyab-embed -m Qwen3-Embedding-0.6B-f16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json
    liyab-embed -m Qwen3-Reranker-0.6B-BF16.gguf --cases tests/data/pooling_cases.json --ref pooling.ref.json

The output is JSON: the token ids of every input (liyab-embed checks its own
against them first), the vectors and the scores.

Needs torch and transformers (pip install torch transformers).
"""

import argparse
import json
from pathlib import Path

# Qwen3-Reranker's model card: the prompt around each formatted pair.
RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                 "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RERANK_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
MAX_LENGTH = 8192  # the cards' max_length


def embedding_modules(model_id):
    """Read the declared pooling and identity Dense modules, independently of the engine."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def path(name):
        return str(Path(model_id) / name) if Path(model_id).is_dir() else hf_hub_download(model_id, name)

    try:
        modules = json.loads(Path(path("modules.json")).read_text())
    except (OSError, ValueError):
        return False, []
    mean, dense = False, []
    for module in modules:
        kind = module["type"].rsplit(".", 1)[-1]
        prefix = module["path"]
        if kind == "Pooling":
            config = json.loads(Path(path(f"{prefix}/config.json")).read_text())
            mean = config.get("pooling_mode_mean_tokens", False)
        elif kind == "Dense":
            config = json.loads(Path(path(f"{prefix}/config.json")).read_text())
            if config.get("bias", False) or not config.get("activation_function", "").endswith(".Identity"):
                raise ValueError("reference supports only bias-free identity Dense modules")
            dense.append(load_file(path(f"{prefix}/model.safetensors"))["linear.weight"].to(torch.float32))
    return mean, dense


def embeddings(model_id, texts, dtype, check_batched, sentence_transformers):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, dtype=getattr(torch, dtype)).eval()
    mean_pooling, dense = embedding_modules(model_id)

    def project(vector):
        for weight in dense:
            vector = F.linear(vector, weight)
        return vector
    out = {"embedding_model": model_id, "embed_ids": [], "embed": [], "embed_mean": [], "embed_last_raw": []}
    with torch.no_grad():
        for text in texts:
            batch = tokenizer([text], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            hidden = model(**batch).last_hidden_state[0].float()  # [tokens, n_embd]
            out["embed_ids"].append(batch["input_ids"][0].tolist())
            last = project(hidden[-1])
            mean = project(hidden.mean(dim=0))
            out["embed_last_raw"].append(last.tolist())
            out["embed"].append(F.normalize(mean if mean_pooling else last, p=2, dim=0).tolist())
            out["embed_mean"].append(F.normalize(mean, p=2, dim=0).tolist())
        if check_batched:
            # The card's own form: one left-padded batch, last position pooled.
            batch = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            hidden = model(**batch).last_hidden_state.float()
            if mean_pooling:
                mask = batch["attention_mask"].unsqueeze(-1)
                hidden = (hidden * mask).sum(1) / mask.sum(1)
            else:
                hidden = hidden[:, -1]
            batched = F.normalize(project(hidden), p=2, dim=1)
            single = torch.tensor(out["embed"])
            cos = (batched * single).sum(dim=1)
            print(f"batched (left padding) vs one at a time: min cosine {cos.min().item():.7f}")
    if sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers is not installed: skipped")
        else:
            # In the reference's precision: sentence-transformers would load the checkpoint's (bfloat16).
            st = SentenceTransformer(model_id, device="cpu", model_kwargs={"dtype": getattr(torch, dtype)})
            encoded = torch.tensor(st.encode(texts, batch_size=1, normalize_embeddings=True))
            cos = (encoded * torch.tensor(out["embed"])).sum(dim=1)
            print(f"sentence-transformers vs transformers: min cosine {cos.min().item():.7f}")
    return out


def rerank_scores(model_id, cases, dtype, sentence_transformers):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix_tokens = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
    out = {"reranker_model": model_id, "rerank_ids": [], "rerank": []}
    with torch.no_grad():
        for case in cases:
            scores, ids = [], []
            for document in case["documents"]:
                pair = f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {case['query']}\n<Document>: {document}"
                tokens = tokenizer([pair], padding=False, truncation="longest_first", return_attention_mask=False,
                                   max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens))["input_ids"][0]
                tokens = prefix_tokens + tokens + suffix_tokens
                logits = model(torch.tensor([tokens])).logits[0, -1].float()
                pair_logits = torch.stack([logits[token_false_id], logits[token_true_id]])
                scores.append(torch.log_softmax(pair_logits, dim=0)[1].exp().item())
                ids.append(tokens)
            out["rerank"].append(scores)
            out["rerank_ids"].append(ids)
    if sentence_transformers:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            print("sentence-transformers is not installed: skipped")
        else:
            # Its CrossEncoder scores a pair by the "yes" and "no" logits (the
            # model's LogitScore module, no activation): the card's P(yes) is
            # the sigmoid of that score.
            st = CrossEncoder(model_id, device="cpu", model_kwargs={"dtype": getattr(torch, dtype)})
            worst = 0.0
            for case, expected in zip(cases, out["rerank"]):
                raw = st.predict([(case["query"], d) for d in case["documents"]], batch_size=1)
                for score, card in zip(torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).tolist(), expected):
                    worst = max(worst, abs(score - card))
            print(f"sentence-transformers vs the card's scores: largest |diff| {worst:.2e}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", help="embedding model: Hugging Face id or local directory")
    parser.add_argument("--reranker", help="reranker: Hugging Face id or local directory")
    parser.add_argument("--cases", required=True, help="cases file (tests/data/pooling_cases.json)")
    parser.add_argument("--out", required=True, help="reference file to write (JSON)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                        help="precision of the reference passes (default float32)")
    parser.add_argument("--check-batched", action="store_true",
                        help="also run the embedding card's left-padded batch and print its distance")
    parser.add_argument("--sentence-transformers", action="store_true",
                        help="also run sentence-transformers (when installed) and print its distance")
    args = parser.parse_args()
    if not args.embedding and not args.reranker:
        parser.error("give --embedding, --reranker or both")

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    out = {}
    if args.embedding:
        out.update(embeddings(args.embedding, cases["embed"], args.dtype, args.check_batched,
                              args.sentence_transformers))
    if args.reranker:
        out.update(rerank_scores(args.reranker, cases["rerank"], args.dtype, args.sentence_transformers))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
