"""Load the base model at a given numerical precision, and generate from it.

SPEC §3: bitsandbytes via transformers. NF4 for 4-bit, LLM.int8 for 8-bit.
SPEC §7: the exact quantization config must land in the results metadata.
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRECISIONS = ("fp16", "8bit", "4bit")

_BYTES_PER_PARAM = {"fp16": 2.0, "8bit": 1.0, "4bit": 0.5}


def quant_config_metadata(precision: str) -> dict:
    """The blob recorded per stage in run metadata (SPEC §7)."""
    if precision == "fp16":
        return {
            "precision": "fp16",
            "method": "none",
            "group_size": None,
            "compute_dtype": "float16",
        }
    if precision == "8bit":
        return {
            "precision": "8bit",
            "method": "bitsandbytes-llm.int8",
            "group_size": None,  # int8 is per-channel, not grouped
            "compute_dtype": "float16",
            "llm_int8_threshold": 6.0,
        }
    if precision == "4bit":
        return {
            "precision": "4bit",
            "method": "bitsandbytes-nf4",
            "group_size": 64,  # bnb NF4 block size
            "compute_dtype": "float16",
            "double_quant": True,
        }
    raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")


def _bnb_config(precision: str):
    if precision == "fp16":
        return None
    if precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unknown precision {precision!r}")


def load_tokenizer(model_id: str):
    """Tokenizer configured for decoder-only *batched* generation (SPEC §6)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    # Left padding is mandatory: with right padding a decoder-only model
    # generates from the pad tokens and the output is garbage.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id: str, precision: str, device: str = "cuda:0"):
    """Load `model_id` at `precision`. Returns (model, tokenizer)."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")

    tok = load_tokenizer(model_id)
    kwargs = {"dtype": torch.float16}
    qcfg = _bnb_config(precision)
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def weight_footprint_mb(model, precision: str) -> float:
    """params x bytes-per-param, in MB (SPEC §7).

    Computed analytically rather than read off the allocator so it is
    comparable across precisions and independent of activation memory.
    """
    n_params = sum(p.numel() for p in model.parameters())
    # bnb stores 4-bit weights packed two-per-byte, so .numel() already
    # reflects the packed count; use the nominal rate against the base
    # parameter count instead.
    return n_params * _BYTES_PER_PARAM[precision] / (1024 ** 2)


def unload(model) -> None:
    """Drop a model and return its VRAM (SPEC §6: one model resident at a time)."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.inference_mode()
def generate_batch(
    model,
    tok,
    messages_list: list[list[dict]],
    max_new_tokens: int,
    batch_size: int = 1,
) -> list[dict]:
    """Greedy-decode a list of chat conversations.

    Greedy (do_sample=False) is deliberate: the experiment compares precisions,
    so decoding must contribute no variance of its own.

    !!! NO CONSTRAINED DECODING. EVER. (SPEC §12) !!!

    Do not add `outlines`, `guidance`, `lm-format-enforcer`, a JSON grammar,
    `prefix_allowed_tokens_fn`, a LogitsProcessor that masks tokens, constrained
    beam search, or any structured-output mode. They would drive the parse-failure
    rate to zero by construction and delete the paper's mechanism evidence — every
    run would look identical on the secondary metric. If parse failures are high,
    that is a result to report, not a bug to engineer away.

    The only arguments this call may ever grow are ones that do not constrain the
    token distribution.

    Returns one dict per input with keys:
        raw_output, hit_token_cap, prompt_tokens, output_tokens, latency_s
    """
    results: list[dict] = []
    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start : start + batch_size]
        texts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]

        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen = out[:, in_len:]
        per_item = elapsed / len(chunk)
        for row, attn in zip(gen, enc["attention_mask"]):
            # Trim trailing pad/eos so output_tokens reflects real generation.
            ids = row.tolist()
            n_gen = len(ids)
            while n_gen > 0 and ids[n_gen - 1] in (tok.pad_token_id, tok.eos_token_id):
                n_gen -= 1
            hit_cap = len(ids) == max_new_tokens and n_gen == len(ids)
            results.append(
                {
                    "raw_output": tok.decode(row, skip_special_tokens=True),
                    "hit_token_cap": hit_cap,
                    "prompt_tokens": int(attn.sum().item()),
                    "output_tokens": n_gen,
                    "latency_s": round(per_item, 4),
                }
            )
    return results
