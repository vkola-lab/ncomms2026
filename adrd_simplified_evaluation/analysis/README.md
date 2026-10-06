# Analysis scripts

Figures, tables and statistics for the LUNAR evaluation. Every figure has its own
script; all loading, statistics and plotting code lives in `common/`.

## Setup

Uses the same environment described [here](https://github.com/yourusername/your-repo/blob/main/adrd_simplified_evaluation/README.md) (`venv_gpu`).

## Running

Run every script from inside this folder, so that `common` can be imported and the relative output paths resolve:

```
cd analysis
python cog.py
```

Results are read from `BASE_RESULTS` in `common/config.py`. Outputs are written
to `../figures_main` (relative to this folder).

## Folder layout

```
analysis/
  common/
    config.py           results path, model names, N_BOOT, N_PERM, plot style
    loading.py          loaders for all result types; EMA smoothing for W&B logs
    metrics.py          per-class, macro and macro-accuracy metrics; label encoding
    bootstrap.py        stratified row-level bootstrap (per-class, macro,
                        macro accuracy, training curves, split baselines)
    permutation.py      paired ID-level permutation tests, alignment checks, BH
    reporting.py        significance stars, p-value matrix, LaTeX tables
    plots.py            figure functions (bars, points, circular, biomarker forest
                        plot, benchmark bars, half-violins)
    plots_training.py   training-curve and training-dynamics figures
  <figure scripts>.py   settings at the top + main()
  entropy/
    entropy_main.py calls `entropy_generate.py` generate responses and token entropies for all models (standalone)
    entropy_generate.py generates responses and token entropies for one model
```

## Scripts

| Script | Reads | Writes |
|---|---|---|
| `cog.py` | `<cohort>/test_cog` for NACC, ADNI, BrainLat, NIFD, PPMI | `../figures_main/fig3_cog_stat_id_level{,_f1}.pdf`, `../figures_main/fig3_cog_table.tex` |
| `etpr.py` | `<cohort>/test_etpr` for NACC, ADNI, BrainLat, NIFD, PPMI | `../figures_main/fig3_macro_circular_bar_plot_sidebyside_id_level{,_f1}.pdf`, `../tables/fig3_etpr_macro_table.tex` |
| `np_one.py` | `NACC/test_np_one` | `../figures/fig3_np_one_{f1,prec_rec}.pdf`, `../figures/fig3_np_one_table.tex` |
| `biomarker.py` | `NACC`, `ADNI` `test_pet` / `test_csf`; `NACC`, `PPMI` `test_dat` | `../figures/forest_biomarkers_macro_{f1,precision,recall}_idlevel.pdf`, `../figures/fig3_biomarker_{pet,csf,dat}_table.tex` |
| `benchmarks.py` | `standard_benchmarks/` | `../figures_main/fig2_benchmarks_macro_perm_id.pdf` |
| `training_curve.py` | `training_curve/` | `../figures_main/fig2_test_perf_over_training_trial_bootstrap.pdf`, `sup_test_perf_over_training_trial_bootstrap_{ablations,sep_benchmarks}.pdf` |
| `output_length.py` | parquet files with `generated_text`, or `token_counts.csv` | `token_counts.csv`, `../figures_main/fig2_output_length_all.pdf` |
| `entropy_plot.py` | `entropy/entropies/*_summary.json` | `../figures_main/fig2_entropy_distribution.pdf` |
| `train_dynamics.py` | `wandb/wandb_export_{reward,reward_std,entropy,mean_length}.csv` | `../figures/fig2_train_plot_bigger.pdf` |

<!-- ### Settings you are likely to change

Each script's settings are at the top of the file.

- **Models:** `MODEL_MAP` (via `model_subset(...)`) selects the models and their
  order. Display names come from `ALL_MODELS` in `common/config.py`. In
  `biomarker.py`, `MODEL_MAP` sets the models in the statistics and LaTeX tables and
  `PLOT_MODELS` the models drawn in the forest plot.
- **Classes:** `CALSS_MAP` sets the display names for the plots in classwise tasks.
- **Table only:** `TABLE_ONLY = True` stops after the bootstrap and the LaTeX table
  (skips permutation tests and figures).
- **Parallelism:** `N_JOBS`, `N_JOBS_BOOT`, `N_JOBS_PERM`.
- **Output:** `OUTPUT_FIG_DIR`, figure names, `LATEX_OUTPUT_PATH`, figure sizes.
- **Output length:** `COMPUTE_TOKEN_COUNTS = True` tokenizes the parquet files and
  saves `token_counts.csv`; `False` reads the saved CSV.
- **Training curve:** `EXCLUDED_BENCHMARKS`, `ABLATION_RUNS`, `PER_TASK_BENCHMARKS`. -->

## Entropy generation

`entropy/entropy_generate.py` runs one model with vLLM on a fixed question subset and saves per-token entropies. Run it from inside `entropy/`, once per model:

```
cd entropy
python entropy_generate.py --model_id <model or checkpoint> --save_name entropies/<key>.json
```

`entropy/entropy_main.py` runs `entropy_generate.py` for all the models. Run it from inside `entropy/` once:

```
cd entropy
python entropy_main.py
```

- The first run samples up to 1,000 questions per dataset and saves them to `random_test_data.csv`; every later run reuses that file, so all models see the same questions. Create it once before starting parallel runs. Delete it (and rerun all models) only if the source data changes.
- Each run writes `<key>.json` (per-token entropies) and `<key>_summary.json` under `intermediate_files/entropies`
  (per-question mean, min, max and question index), which `entropy_plot.py` reads.
- `<key>` must match a key in `KEY_NAMES` in `entropy_plot.py`
  (`q3b`, `q7b`, `oversample`, `oversample_sce_tanh`, `oversample_dedup`, `oversample_dedup_sce_tanh`).

## Statistical methods

- **Point estimates** are always computed on the original data.
- **Bootstrap:** 1,000 row-level resamples; 95% percentile interval. Rows are resampled within strata so their sizes stay fixed: within each cohort for the
  pooled "Other" group (COG), within each benchmark (standard benchmarks), within each cohort x benchmark cell (training curves).
- **Per-class metrics** (COG, NP-one): precision, recall and F1 for each class present in the ground truth. Resamples in which a class has no true cases are
  dropped from that class's interval; the script prints a warning if this happens.
- **Macro metrics** (ETPR, biomarker): unweighted mean over the classes present in the cohort's ground truth. A class absent from a resample is left out of that
  resample's average. 
- **Standard benchmarks and training curves:** accuracy per benchmark (or per cohort x benchmark cell), then averaged with equal weight.
- **Permutation tests:** paired, 10,000 permutations. For each question/patient ID the two models' results are swapped with probability 1/2 (all repeats of an ID
  together). p = (number of |null difference| >= |observed difference| + 1) / (10,000 + 1). IDs are prefixed with cohort or benchmark, because the same ID can
  occur in different datasets. Scripts stop with an error if two models do not have the same IDs, rows or ground truth.
- **Output length:** tested on per-question mean log10 token count (repeats averaged first); effects are reported as length ratios. Truncated responses are
  included; truncation rates are printed.
- **Entropy:** mean token entropy per question, from the top-20 token probabilities (renormalised), in nats.
- **Multiple comparisons:** Benjamini-Hochberg across model pairs within each (cohort, class, metric) for per-class results, within each (cohort, metric) for macro results, and across all pairs for benchmarks, output length and entropy.
- **Reproducibility:** all random draws are seeded. Permutations and training-curve bootstrap resamples are drawn in chunks; the results depend on the chunk size
  (`PERM_CHUNK` in `config.py`, `PERM_CHUNK_TOKENS` in `output_length.py`, `chunk` in the training-curve bootstrap), so keep it fixed.

## Adding a new figure

1. Put data loading and preparation in `common/loading.py`, or reuse a loader.
2. Put any statistics in `common/bootstrap.py` or `common/permutation.py`.
3. Put the figure function in `common/plots.py` (or `common/plots_training.py` for curves over training steps), with every figure-specific choice as an argument.
4. Add `<name>.py` in this folder with the settings at the top and a short `main()`.

## Intermediate files

Each script has a flag `LOAD_FROM_ORIGINAL` to load the original results. If this is set to true or if the given `INTERMEDIATE_FILE_PATH` does not exist, the code loads the original result files and saves the loaded file to `INTERMEDIATE_FILE_PATH`. If the it is set to false and the given `INTERMEDIATE_FILE_PATH` exists, the script reads that file instead. All intermediate files are saved under `intermediate_files`.