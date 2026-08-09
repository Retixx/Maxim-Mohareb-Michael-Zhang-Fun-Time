# Gate C evidence — Qwen3-1.7B 4-bit, Kaggle T4

Committed here rather than under `analysis/`, which is git-ignored. These are
the raw artifacts behind `clean_room/CODEX_HANDOFF_BUG12.md`.

Source config: `local_gpu_memory_matched_4bit_v1`, `--model small`,
batch 4, frozen excluded n=200, seed 20260806, run at commit `56cd63f`.

To analyse in place, point the loader at this directory instead of
`analysis/local_smoke/small_4bit_bs4/results/`.

Headline: MA F1 0.2312 vs single-hop 0.4213, dF1 -19.01 pts,
95% CI [-26.58, -11.56], McNemar p=0.00026.
