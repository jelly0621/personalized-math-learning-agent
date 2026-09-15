# Phase 7 Pilot Evaluation

This evaluation separates textbook-derived answers, manual seed labels, model predictions, deterministic metrics, and LLM-judged metrics.

`max_surface_similarity_to_real_seed` is only a normalized text-surface signal. It is not semantic plagiarism detection.

Run deterministic validation without API calls:

```powershell
python -m scripts.run_phase7_eval --dry-run
```

Run the bounded real pilot manually:

```powershell
python -m scripts.run_phase7_eval --suite pilot
```

The Phase 7.1 defaults challenge up to five eligible cases and cap the full
pilot at 50 LLM calls. A case is eligible only when its Solver answer matches
the textbook answer symbolically and the deterministic corruption is proven
not equivalent.
