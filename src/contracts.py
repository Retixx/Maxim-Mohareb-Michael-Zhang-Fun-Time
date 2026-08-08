"""Shared immutable identifiers for repaired experiment artifacts."""

EXPERIMENT_SCHEMA = "open_corpus_marag_v4"
PILOT_GATE_SCHEMA_VERSION = 4
CAMPAIGN_PLAN_SCHEMA_VERSION = 3
ANALYSIS_SCHEMA_VERSION = 4

QWEN3_HYBRID_FAMILY = "Qwen3-hybrid-april-2025"
QWEN3_HYBRID_MODELS = {
    "large": "Qwen/Qwen3-14B",
    "base": "Qwen/Qwen3-8B",
    "mid": "Qwen/Qwen3-4B",
    "small": "Qwen/Qwen3-1.7B",
    "tiny": "Qwen/Qwen3-0.6B",
}
QWEN3_THINKING_MODE = False


def model_policy_identity() -> dict:
    """Return the canonical artifact identity for the model-family axis."""
    return {
        "name": QWEN3_HYBRID_FAMILY,
        "models": dict(QWEN3_HYBRID_MODELS),
    }


def validate_model_contract(
    config: dict, *, allow_local_smoke: bool = False
) -> None:
    """Enforce one switchable Qwen3 checkpoint family and non-thinking mode."""
    expected_aliases = {
        alias: model_id
        for alias, model_id in QWEN3_HYBRID_MODELS.items()
        if alias != "base"
    }
    configured_aliases = config.get("models")
    if configured_aliases != expected_aliases:
        raise ValueError(
            "models must be the exact Qwen3 hybrid size catalog; "
            f"configured={configured_aliases!r} expected={expected_aliases!r}"
        )

    smoke = config.get("local_smoke") or {}
    if smoke:
        if not allow_local_smoke:
            raise ValueError("local smoke model selection is not valid for production")
        active_alias = smoke.get("model_alias")
        if active_alias not in {"tiny", "small"}:
            raise ValueError("local smoke must select the tiny or small Qwen3 hybrid model")
        expected_base = QWEN3_HYBRID_MODELS[active_alias]
        if smoke.get("model_id") != expected_base:
            raise ValueError("local smoke model identity is outside the Qwen3 hybrid catalog")
    else:
        expected_base = QWEN3_HYBRID_MODELS["base"]
    if config.get("model_id") != expected_base:
        raise ValueError(
            "model_id must select the contracted Qwen3 hybrid base; "
            f"configured={config.get('model_id')!r} expected={expected_base!r}"
        )

    if config.get("thinking_mode") is not QWEN3_THINKING_MODE:
        raise ValueError("thinking_mode must be the literal boolean false")

    approved_ids = set(QWEN3_HYBRID_MODELS.values())
    revisions = config.get("model_revisions")
    if not isinstance(revisions, dict) or set(revisions) != approved_ids:
        raise ValueError("model_revisions must cover the exact Qwen3 hybrid catalog")
    tokenizer_revisions = config.get("tokenizer_revisions")
    if tokenizer_revisions is not None and (
        not isinstance(tokenizer_revisions, dict)
        or set(tokenizer_revisions) != approved_ids
    ):
        raise ValueError("tokenizer_revisions must cover the exact Qwen3 hybrid catalog")

    approved_names = {"base", *expected_aliases}
    for run_id, definition in (config.get("runs") or {}).items():
        if not isinstance(definition, dict):
            raise ValueError(f"run {run_id!r} must be a stage mapping")
        for stage, spec in definition.items():
            if not isinstance(spec, dict):
                continue
            model_name = spec.get("model", "base")
            if model_name not in approved_names:
                raise ValueError(
                    f"run {run_id!r} stage {stage!r} selects {model_name!r}; "
                    "only Qwen3 hybrid size aliases are allowed"
                )
