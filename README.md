# Project C — Customer Support QLoRA

Project execution is governed by `PROJECT_C_QLORA_MASTER_PLAN_v1.1.md`.

## Stage status

- Stage C1 — Dataset Analysis: complete (PASS). Artifacts are in `artifacts/stage1/` and the report is `reports/stage1_dataset_analysis.md`. Re-run with `.venv/bin/python -m src.data.analyze_dataset`.
- Stage C2 — Task Definition & Evaluation Contract: complete (PASS). Configs are in `configs/`; reports are `reports/task_definition.md` and `reports/evaluation_contract.md`. Rebuild from Stage C1 artifacts with `.venv/bin/python -m src.evaluation.build_contracts`.
- Stage C3 — Frozen Dataset Construction: complete (PASS). Frozen files are in `data/processed/`; manifests are in `data/manifests/`; validation artifacts are in `artifacts/stage3/`. Rebuild with `.venv/bin/python -m src.data.build_frozen_splits`.
- Stage C4 and later: not started.

Stages C1–C2 do not create Train, Validation, Dev, or Locked Test splits and do not load, train, or evaluate models.
