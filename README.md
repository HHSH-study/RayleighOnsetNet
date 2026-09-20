# RayleighOnsetNet

Frozen-model inference and supporting data for **RayleighOnsetNet: Physics-Motivated Two-Stage Deep Learning for Long-Period Rayleigh-Wave Onset Picking in Sedimentary Basins**, by Haishun Huang, Hanling Wang, Hongwei Wang and Ruizhi Wen. Correspondence: Hongwei Wang, whw1990413@163.com.

This is an inference release, not a full training repository. The two checkpoint files require both `rayleigh_onsetnet_model.py` (network definition) and `predict.py` (input handling and coarse-to-refine inference).

## 1. Run the included earthquake record

Open a terminal **inside this repository folder**. In a separate Python environment, install the dependencies and run:

```sh
python -m pip install -r requirements.txt
python predict.py --input example/NZ_2016_Kaikoura_WEL_20.v --output example_prediction.csv --device cpu
```

The included example is a processed GeoNet WEL station record of the 2016 Kaikōura earthquake: 30,000 samples at 100 Hz, spanning 0–299.99 s. The expected output is:

| File | Coarse onset (s) | Refined onset (s) | Status |
| --- | ---: | ---: | --- |
| NZ_2016_Kaikoura_WEL_20.v | 105.741 | 105.613 | ok |

These are **model predictions**, measured from the first input sample, not earthquake origin time. Small last-digit differences can occur across hardware/software. The example is already part of the external 142-record table; it is not an additional evaluation sample. Its analyst reference is 102.058 s (class B), illustrating that a model prediction and a reference need not coincide.

Tested on Windows, Python 3.14.2, NumPy 2.4.3 and PyTorch 2.11.0+cpu. These are release-test versions, not a claim about the original training environment. CUDA is optional (`--device cuda`) and requires a compatible PyTorch/GPU installation; the release test used CPU. Existing output files are not overwritten: use a new output name when rerunning.

## 2. Use your own record or a batch

```sh
python predict.py --input my_record.v --output my_prediction.csv --device cpu
python predict.py --input my_velocity_folder --output batch_predictions.csv --device cpu
```

The folder command processes all `.v` files directly in that folder. Each input is a text file: first line `dt_s npts`, then `npts` rows of **H1, H2, UD velocity in cm/s**. For example, the included file begins with `1.00000000E-02 30000`. Only the third (vertical) column is used for inference.

Prepare physical velocity at **100 Hz** from the source data before running the command. The script does not remove instrument response, integrate acceleration or resample input. It reads the full supplied record; it does not require padding/cropping to the model-development length. Per-record normalization, the model's 2–10 s CWT and the 40 s refinement window are handled internally; do not pre-normalize or add a task-specific band-pass filter to imitate these steps.

The output CSV contains `filename`, `npts`, `sampling_rate_hz`, `coarse_s`, `refined_s` and `status`. A status of `outside_record` flags an untrimmed refined prediction outside the supplied time range. The original EMA checkpoint weights are used for both stages.

## 3. Supporting study data

| File under `data/` | Contents |
| --- | --- |
| `internal_records_770.csv` | Development metadata, reference onsets, 617 training / 153 hold-out assignments, and two-stage predictions. |
| `external_records_142.csv` | All 142 external records: 56 A, 66 B and 20 C, with references and frozen-model predictions. |
| `external_record_metadata_142.json` | Source identifiers, coordinates, sampling, units, time references, processing and input checksums. |
| `fig7_region_summary.csv` | Five-region counts and refined timing results for Fig. 7. |
| `discussion_case_predictions_33.csv` | Predictions for the independent Japanese case event; no analyst reference labels. |
| `discussion_case_record_ids_33.json` | Source identifiers and processed-input checksums for those 33 records. |
| `window_sensitivity_records_44.csv` | Fig. 11/Table 7 derived metrics: 44 screened records, including the 35-record primary subset, at five imposed shifts. |

Key fields and conventions:

- Internal: `True_Time_s` is the reference; `Coarse_Pred_s` / `Final_Pred_s` are predictions; `Subset` identifies training or hold-out. `Trace_Name` is the source trace identifier. The 153-record hold-out subset also monitored checkpoint selection.
- External: `human_s` is the analyst reference; `abc` is the class; `coarse_s` / `refined_s` are predictions. Class C has no reference to score but remains in the coverage denominator. A+B provides 122 evaluable records. Join `Filename` to `filename` in the metadata JSON; inherited case-sensitive duplicate identifiers agree.
- A denotes a clear, defensible single onset; B an identifiable but gradual/mixed boundary; C no unique defensible boundary or unsuitable for the 2–10 s task. The external set comprises representative cases from eight earthquakes in five regions, not a random global test.
- Case event: join `Filename` in the CSV to `filename` in its JSON. Prediction columns are `Coarse_sec` and `Refine_sec`.
- Times/errors are in seconds from the processed record's first sample. Signed error is prediction minus reference. `event_time` is an event identifier, not waveform zero time. Coordinates are degrees; depth and distance are km. Missing metadata remain missing.
- Acc@δ = 100 × fraction with absolute error ≤ δ. MAE and Median AE summarize absolute errors. Regional accuracies use A+B, while `total_n` includes C.
- Sensitivity: join `trace_name` to internal `Trace_Name`. `primary_long_mixed=True` selects 35 records from 16 events. There are 220 rows (44 records × −10, −5, 0, +5, +10 s). `source_duration_s` is pre-padding duration. `reference_s` is the analyst onset; `reference_grid_s` is sample-aligned; `fixed_end_s` stays fixed. `shift_s` is an imposed start shift, **not a model error**.
- The three sensitivity columns are percentage-valued: signed 2–10 s squared-velocity-integral change (`energy_change_pct`), Konno–Ohmachi-smoothed velocity FAS relative L2 difference (`fas_smoothed_l2_pct`, b=40), and 5%-damped windowed PSA relative L2 difference (`psa_l2_pct`). All values, including those outside plotted axes, remain in the table. Original settings: 0.01 s sampling; 0.1–0.5 Hz filtering only for the integral; `dt*abs(rfft)`, 131072 FFT points and a 2 s end taper; PSA periods 2–10 s in 0.1 s steps and 100 s free decay. Reported sensitivity intervals used 5,000 event-cluster bootstrap resamples, seed 20260913.

These tables support checks of the reported results; they do not by themselves regenerate spectra from raw waveforms or retrain the model. Record-level numerical results and checkpoint tensors have not been altered for this release.

## 4. Source archives and the example's attribution

Obtain original study waveforms under the respective archive terms:

- Japanese development and independent case event: [NIED K-NET/KiK-net](https://doi.org/10.17598/NIED.0004).
- Taiwan CWB records (36): PEER NGA/COSMOS [Virtual Data Center](https://www.strongmotioncenter.org/vdc/scripts/default.plx).
- Los Angeles Basin, El Mayor–Cucapah (31) and Fontana (30): [SCEDC](https://doi.org/10.7909/C3WD3xH1).
- Gorkha, two events (3 + 3): Hokkaido University–Tribhuvan University Kathmandu Valley array on [Figshare](https://doi.org/10.6084/m9.figshare.19809052).
- Kaikōura/Wellington (13): [GeoNet seismic waveform dataset](https://doi.org/10.21420/G19Y-9D40).
- Emilia, two events (11 + 15): [ESM v3.0](https://doi.org/10.13127/esm.3).

Only **one processed GeoNet example** is redistributed. Source: NZ.WEL.20, HNE/HNN/HNZ channels, record start 2016-11-13 11:02:46 UTC. Processing was instrument-sensitivity removal (not full response deconvolution), linear detrending, 0.05 Hz high-pass filtering, integration to velocity, conversion to cm/s and resampling from 200 to 100 Hz. The existing processed study file is supplied without further changes; details and SHA-256 are in the external metadata JSON.

The example is adapted from GeoNet data, copyright Earth Sciences New Zealand, under [Creative Commons Attribution 3.0 New Zealand](https://creativecommons.org/licenses/by/3.0/nz/), as specified by the [GeoNet data policy](https://www.geonet.org.nz/policy). We acknowledge GeoNet and its sponsors NHC, Earth Sciences NZ, LINZ, NEMA and MBIE. No endorsement is implied. No other third-party waveform files, manuscript drafts or private correspondence are included.

## 5. Citation and reuse

Cite the authors and study title above and identify the repository version/commit used. No article DOI has been assigned in this package.

No open-source/open-data license has yet been assigned to the study code, checkpoints or study-generated tables. Contact the corresponding author for reuse permission. The GeoNet example retains its separate attribution license; that license does not apply to the authors' code or weights.
