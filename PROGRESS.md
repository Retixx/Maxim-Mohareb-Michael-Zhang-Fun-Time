# Current handoff

This file records only the active final-campaign state. Detailed history of the
earlier 1.5B/T4/Kaggle tiers remains available in Git history and must not be
treated as part of the final n=1,500 analysis.

## 2026-08-05 — final A100 contract frozen; CPU verification passed

### Human-approved design

- One n=1,500 HotpotQA distractor/validation sample is shared by every arm.
- F1 is primary; EM is co-reported.
- 3B 8-bit versus 1.5B FP16 is the primary near-memory-matched Q-vs-S treatment.
- 3B 4-bit remains a separate mandatory secondary treatment.
- All five 0.5B FP16 runs remain as an appendix-only compliance/capacity floor.
  They are excluded from the selector and main-paper allocation claims.
- The static matrix is the prior 21 arms plus `single_fp16`: 22 runs.
- The post-ablation selector considers only 3B FP16/8-bit/4-bit and 1.5B FP16,
  then adds at most one distinct exploratory run. Campaign maximum: 23.
- Optimized-role results are explicitly in-sample/exploratory.
- Batch size is pinned at 32; production OOMs fail rather than autotune.
- One uncontended A100 provides standardized timing ratios against uniform
  four-role 3B FP16 = `1.00x`; model loading is co-reported separately.

### Seed correction completed

The final sample was generated from the pinned dataset revision and committed at
`config/manifests/final_n1500_seed20260805.json`.

```text
n                              1500
seed                           20260805
unique excluded IDs            3031
final/exclusion overlap        0
manifest file SHA-256          841dbca9ac7e76c0277a5696fba9f7e254b973afb0b670efbc5edfc006b4af46
final ordered-ID SHA-256       5d4cc24872aeb603cbd005f790958199ef4cc993a1e7f048403608603da602af
sorted exclusion-ID SHA-256    a5cfacb84fa9a48217f3206a095706a6d48802bd244151c72f2eef08372c00a8
```

The exclusion union contains the exact 3,000 IDs from the old result blob, the
exact 30 committed seed-1234 development IDs, and reconstructed seed-0/n=10
development IDs. The manifest records provenance and all exact IDs.

### Memory contract

Measured resident weight footprints:

```text
3B FP16       5886.0 MiB
3B 8-bit      3240.0 MiB
1.5B FP16     2944.4 MiB
3B 4-bit      1917.0 MiB
0.5B FP16      942.3 MiB
```

The primary treated-model match differs by 295.6 MiB: 3B 8-bit holds 10.04%
more. With the shared 3B FP16 configuration included in a one-role system, the
gap is 3.35%. Paper wording: **near-memory-matched**.

Primary accounting is deduplicated concurrent model-footprint MiB (parameters
plus buffers, one charge per exact configuration). Sequential peak VRAM,
per-role-service totals, parameters, buffers, and cold loading are separate.

Timing uses two repetitions of a frozen 128-question excluded cohort on one
reserved A100. It is disjoint from the final 1,500 and warm-up 32, never enters
accuracy/selection, and excludes all five 0.5B floor arms.

### Active files

- `SPEC.md`: final experiment/claim contract
- `config/experiment.yaml`: 22-run static matrix plus 256-candidate selector
- `config/manifests/final_n1500_seed20260805.json`: final and excluded IDs
- `scripts/freeze_final_sample.py`: auditable manifest generator
- `README.md`: current execution handoff

Historical notebooks, committed interim outputs, and `gate2_report.py` have been
removed from the active tree. Git history retains them.

### Local verification

The CPU-only integrity suite passes 31/31 tests, including frozen cohorts,
salvage propagation, solo scoring, canonical resume, selector constraints,
timing isolation, campaign planning, and final completeness checks.

### A100 gate still required before final accuracy

1. Runner validates the final and exclusion manifest, hashes, dataset revision,
   order, and zero overlap before model load.
2. Model/tokenizer revisions are resolved and recorded exactly.
3. Salvaged Extractor evidence reaches QA without changing original parse status.
4. Canonical batch membership survives interruption/resume.
5. Every distinct model/stage path passes batch 32 on the production A100 stack.
6. `single_fp16` completes through the dedicated solo path and uses a fair frozen
   prompt and the same answer budget/context.
7. Metadata contains environment, batch, token, load-time, timing, and complete
   memory-accounting fields.
8. `analyze.py` dry-run verifies question-paired F1 inference, Holm correction,
   batch/question clustering, the conservative 95%-lower-bound 256-allocation
   selector, frozen output artifacts, and exploratory label.
9. One A100 environment artifact freezes the container digest and full package/
   driver/GPU lock; all production workers match it.
10. Static run count is exactly 22; a dynamic selected run can make at most 23.

If any check fails, fix and repeat preflight on excluded development data. Do not
consume final-manifest outputs while debugging.
