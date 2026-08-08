"""Load the base model at a given numerical precision, and generate from it.

SPEC §3: bitsandbytes via transformers. NF4 for 4-bit, LLM.int8 for 8-bit.
SPEC §7: the exact quantization config must land in the results metadata.
"""

import gc
import hashlib
import json
import time

import torch

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
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "quantized inference requires transformers and bitsandbytes; "
            "install the pinned experiment environment before A100 preflight"
        ) from exc
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


def load_tokenizer(model_id: str, revision: str | None = None):
    """Tokenizer configured for decoder-only *batched* generation (SPEC §6)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "inference requires transformers; install the pinned experiment environment"
        ) from exc
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # Left padding is mandatory: with right padding a decoder-only model
    # generates from the pad tokens and the output is garbage.
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_id: str,
    precision: str,
    device: str = "cuda:0",
    revision: str | None = None,
    tokenizer_revision: str | None = None,
):
    """Load `model_id` at `precision`. Returns (model, tokenizer)."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "inference requires transformers; install the pinned experiment environment"
        ) from exc
    tok = load_tokenizer(model_id, revision=tokenizer_revision or revision)
    kwargs = {"dtype": torch.float16}
    qcfg = _bnb_config(precision)
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **kwargs)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def weight_footprint_mib(model, precision: str = None) -> float:
    """Actual parameter tensor bytes, in binary MiB (SPEC §7).

    Summed per tensor as numel x element_size, which is exact at every precision
    and needs no special-casing: bitsandbytes stores 4-bit weights as uint8
    (element_size 1) and int8 weights as int8 (element_size 1), while everything
    it leaves alone stays fp16 (element_size 2).

    Do NOT reintroduce the `total_params * bytes_per_param` shortcut. A
    Params4bit tensor's .numel() is the PACKED byte count — two 4-bit values per
    uint8 — so multiplying it by 0.5 bytes/param applies the compression twice.
    That bug understated the 4-bit footprint by 2.6x (423.7 MB reported against a
    true 1070.2 MB), and this number is reported in the paper.

    Note that this is genuinely mixed-precision: bnb quantizes the linear layers
    but leaves embeddings, biases and norms at fp16. For Qwen2.5-1.5B that is
    1.31B params at 4-bit plus 233M still at fp16 — which is why a "4-bit" stage
    costs 1070 MB rather than the 772 MB a uniform 4-bit model would.

    Cross-checked against transformers' model.get_memory_footprint(): exact match.
    """
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def weight_footprint_mb(model, precision: str = None) -> float:
    """Deprecated compatibility alias; the returned binary unit is MiB."""
    return weight_footprint_mib(model, precision)


def memory_footprint_mib(model) -> dict:
    """Return binary-MiB resident tensor census, including model buffers."""
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    fallback = parameter_bytes + buffer_bytes
    try:
        full_bytes = int(model.get_memory_footprint(return_buffers=True))
    except (AttributeError, TypeError):
        full_bytes = fallback
    # Never let a library-version difference omit known resident buffers.
    full_bytes = max(full_bytes, fallback)
    mib = 1024 ** 2
    return {
        "parameter_mib": parameter_bytes / mib,
        "parameter_bytes_mib": parameter_bytes / mib,
        "buffer_mib": buffer_bytes / mib,
        "buffer_bytes_mib": buffer_bytes / mib,
        "model_footprint_mib": full_bytes / mib,
        "model_footprint_bytes": full_bytes,
    }


def config_fingerprint(
    model_id: str,
    precision: str,
    revision: str | None,
    tokenizer_revision: str | None,
) -> tuple[str, dict]:
    """Complete model/config identity used for resident-weight deduplication."""
    payload = {
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": tokenizer_revision or revision,
        **quant_config_metadata(precision),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def resolved_revision_metadata(
    model,
    tok,
    expected_model_revision: str,
    expected_tokenizer_revision: str,
    *,
    allow_unpinned_tbd: bool = False,
) -> dict:
    """Verify HF resolved commits when the installed library exposes them."""
    resolved_model = getattr(getattr(model, "config", None), "_commit_hash", None)
    resolved_tokenizer = (
        getattr(tok, "init_kwargs", {}).get("_commit_hash")
        or getattr(tok, "_commit_hash", None)
    )
    if (
        resolved_model
        and resolved_model != expected_model_revision
        and not (allow_unpinned_tbd and expected_model_revision == "TBD")
    ):
        raise RuntimeError(
            f"model resolved to {resolved_model}, expected {expected_model_revision}"
        )
    if (
        resolved_tokenizer
        and resolved_tokenizer != expected_tokenizer_revision
        and not (allow_unpinned_tbd and expected_tokenizer_revision == "TBD")
    ):
        raise RuntimeError(
            f"tokenizer resolved to {resolved_tokenizer}, expected {expected_tokenizer_revision}"
        )
    return {
        "resolved_model_revision": resolved_model or expected_model_revision,
        "resolved_tokenizer_revision": resolved_tokenizer or expected_tokenizer_revision,
        "revision_pin_status": (
            "unpinned_smoke_TBD" if allow_unpinned_tbd else "pinned"
        ),
    }


def param_census(model) -> dict:
    """Nominal and actual bitsandbytes parameter counts."""
    nominal = quantized_4bit = quantized_8bit = 0
    for p in model.parameters():
        if p.__class__.__name__ == "Params4bit":
            n = p.numel() * 2  # two 4-bit values per packed uint8
            quantized_4bit += n
            nominal += n
        elif p.__class__.__name__ == "Int8Params" or p.dtype == torch.int8:
            quantized_8bit += p.numel()
            nominal += p.numel()
        else:
            nominal += p.numel()
    return {
        "nominal_params": nominal,
        "quantized_params": quantized_4bit + quantized_8bit,
        "quantized_4bit_params": quantized_4bit,
        "quantized_8bit_params": quantized_8bit,
    }


MIN_QUANTIZED_FRACTION = 0.5


def validate_loaded_precision(model, precision: str, model_id: str) -> dict:
    """Prove requested precision from loaded tensor types before generation."""
    if precision not in PRECISIONS:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of {PRECISIONS}"
        )
    census = param_census(model)
    nominal = int(census["nominal_params"])
    quantized = int(census["quantized_params"])
    if nominal <= 0:
        raise RuntimeError(f"{model_id}: loaded model has no nominal parameters")

    total_fraction = quantized / nominal
    four_bit_fraction = int(census["quantized_4bit_params"]) / nominal
    eight_bit_fraction = int(census["quantized_8bit_params"]) / nominal
    if precision == "fp16":
        requested_fraction = 1.0 - total_fraction
        valid = quantized == 0
        expectation = "zero quantized parameters"
    elif precision == "4bit":
        requested_fraction = four_bit_fraction
        valid = (
            four_bit_fraction >= MIN_QUANTIZED_FRACTION
            and census["quantized_8bit_params"] == 0
        )
        expectation = f"4bit fraction>={MIN_QUANTIZED_FRACTION:.2f} and no 8bit parameters"
    else:
        requested_fraction = eight_bit_fraction
        valid = (
            eight_bit_fraction >= MIN_QUANTIZED_FRACTION
            and census["quantized_4bit_params"] == 0
        )
        expectation = f"8bit fraction>={MIN_QUANTIZED_FRACTION:.2f} and no 4bit parameters"

    if not valid:
        raise RuntimeError(
            f"{model_id}: requested {precision} but loaded tensor census has "
            f"nominal_params={nominal}, quantized_params={quantized}, "
            f"quantized fraction={total_fraction:.6f}, "
            f"4bit fraction={four_bit_fraction:.6f}, "
            f"8bit fraction={eight_bit_fraction:.6f}; expected {expectation}"
        )
    return {
        **census,
        "quantized_fraction": total_fraction,
        "requested_precision_fraction": requested_fraction,
        "precision_validation_passed": True,
    }


def unload(model=None) -> None:
    """Reclaim VRAM (SPEC §6: exactly one model resident at a time).

    **The caller must drop its own reference to the model BEFORE calling this.**
    `del model` here only clears this function's parameter binding; if the caller
    still holds the object it stays reachable, `gc.collect()` cannot collect it,
    and `empty_cache()` frees only unused cached blocks — so the weights survive.

    This mattered: the stage loop used to call `unload(model)` while `model` was
    still bound in the caller, then load the next stage's model into the *same*
    variable. The rebinding is what finally released the old one, which happens
    after `from_pretrained` has already allocated the new one — so both models
    were briefly resident, exactly the thing SPEC §6 exists to prevent. It went
    unnoticed because the sweeps run on a 16 GB T4 with 2.9 GB models.

    Prefer:  model = tok = None; models.unload()
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.inference_mode()
def sequence_confidence(model, tok, enc: dict, gen_ids, n_gen_per_item: list[int],
                        max_chunk: int = 8) -> list[dict]:
    """Per-item confidence over the generated tokens, for the calibration half of
    SPEC §1's mechanism claim (see the §5b prediction 4).

    Returns per item: mean_logprob, min_logprob, mean_entropy — all over that
    item's real generated tokens only, excluding padding.

    Implemented as ONE teacher-forced forward pass over prompt+output, restricted
    to the output positions. Deliberately NOT a LogitsProcessor: SPEC §12 forbids
    constrained decoding, and although an observe-only processor would not violate
    that ban, having no logits hook at all removes the ambiguity entirely. This
    runs strictly AFTER generation and cannot influence what was generated —
    greedy outputs are bit-identical whether or not this is called.

    Costs roughly one extra prefill per batch, which is why it is opt-in
    (`generation.log_confidence` in the config, default false).
    """
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    B, P = input_ids.shape
    G = gen_ids.shape[1]

    gen_mask = torch.zeros((B, G), dtype=attn.dtype, device=attn.device)
    for i, n in enumerate(n_gen_per_item):
        if n > 0:
            gen_mask[i, :n] = 1

    results: list[dict] = []
    for start in range(0, B, max_chunk):
        sl = slice(start, start + max_chunk)
        full = torch.cat([input_ids[sl], gen_ids[sl]], dim=1)
        full_attn = torch.cat([attn[sl], gen_mask[sl]], dim=1)

        # Keep logits only for the positions that predict generated tokens.
        # Materialising all positions would be vocab x seq x batch and blow up.
        logits = None
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                logits = model(full, attention_mask=full_attn, **{kw: G + 1}).logits
                break
            except TypeError:
                continue
        if logits is None:  # older signature: fall back to full logits
            logits = model(full, attention_mask=full_attn).logits[:, P - 1 :, :]

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(-1)  # [b, G+1]

        chunk_gen = gen_ids[sl]
        for i in range(chunk_gen.shape[0]):
            n = n_gen_per_item[start + i]
            if n <= 0:
                results.append({"mean_logprob": None, "min_logprob": None,
                                "mean_entropy": None})
                continue
            idx = chunk_gen[i, :n]
            lp = logprobs[i, torch.arange(n, device=logprobs.device), idx]
            results.append({
                "mean_logprob": round(lp.mean().item(), 5),
                "min_logprob": round(lp.min().item(), 5),
                "mean_entropy": round(entropy[i, :n].mean().item(), 5),
            })
        del logits, logprobs, probs, entropy
    return results


@torch.inference_mode()
def generate_batch(
    model,
    tok,
    messages_list: list[list[dict]],
    max_new_tokens: int,
    batch_size: int = 1,
    log_confidence: bool = False,
    force_full_generation: bool = False,
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
        context_candidates = [
            getattr(getattr(model, "config", None), name, None)
            for name in ("max_position_embeddings", "n_positions", "seq_length")
        ]
        context_candidates.append(getattr(tok, "model_max_length", None))
        context_candidates = [
            int(value) for value in context_candidates
            if isinstance(value, (int, float)) and 0 < value < 1_000_000_000
        ]
        context_window = min(context_candidates) if context_candidates else None
        if context_window is not None and in_len + max_new_tokens > context_window:
            raise RuntimeError(
                "prompt plus frozen output budget exceeds model context window: "
                f"{in_len}+{max_new_tokens}>{context_window}"
            )

        # CUDA kernels are asynchronous. Synchronizing on both sides isolates
        # this batch from work queued by the preceding batch.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": tok.pad_token_id,
        }
        if force_full_generation:
            # Excluded preflight must allocate the worst-case KV cache rather
            # than passing because the model happened to emit EOS early.
            generation_args["min_new_tokens"] = max_new_tokens
        out = model.generate(**enc, **generation_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gen = out[:, in_len:]
        per_item = elapsed / len(chunk)

        # Every id generation may stop on, not just `tok.eos_token_id`.
        # `generation_config.eos_token_id` is a LIST for Llama-3.x (three ids)
        # and for Qwen2.5-Instruct (two). Trimming against the tokenizer
        # attribute alone leaves a trailing stop token on any sequence that
        # terminated on one of the others: `output_tokens` comes out one too
        # high, and in a batch that reached the cap `n_gen == len(ids)` still
        # holds, so `hit_token_cap` is wrongly True and the record is
        # mislabelled `truncated`. Qwen2.5 escaped this only because its second
        # stop id happens to equal the pad id. Model 2 would not have.
        stop_ids = {tok.pad_token_id, tok.eos_token_id}
        cfg_eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple, set)):
            stop_ids |= set(cfg_eos)
        elif cfg_eos is not None:
            stop_ids.add(cfg_eos)
        stop_ids.discard(None)

        n_gen_per_item = []
        rows = []
        for row, attn in zip(gen, enc["attention_mask"]):
            # Trim trailing pad/stop tokens so output_tokens reflects real generation.
            ids = row.tolist()
            n_gen = len(ids)
            while n_gen > 0 and ids[n_gen - 1] in stop_ids:
                n_gen -= 1
            hit_cap = len(ids) == max_new_tokens and n_gen == len(ids)
            n_gen_per_item.append(n_gen)
            rows.append({
                "raw_output": tok.decode(row, skip_special_tokens=True),
                "hit_token_cap": hit_cap,
                "prompt_tokens": int(attn.sum().item()),
                "output_tokens": n_gen,
                "generated_sequence_tokens": len(ids),
                "latency_s": round(per_item, 4),
                "batch_wall_s": elapsed,
                "batch_size_actual": len(chunk),
                "padded_input_tokens": int(in_len),
                "context_window_tokens": context_window,
                "forced_full_generation": force_full_generation,
            })

        if log_confidence:
            # Runs after generation; cannot affect what was generated.
            conf = sequence_confidence(model, tok, enc, gen, n_gen_per_item)
            for r, c in zip(rows, conf):
                r.update(c)

        results.extend(rows)
    return results
