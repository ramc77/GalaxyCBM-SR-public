# Cross-survey findings

Reference sample: **DECaLS (gz_desi)**

## Deltas relative to reference

| survey | n_ref | n_shift | delta_accuracy | delta_macro_f1 | delta_kappa | delta_ece | delta_coverage | rule_mean_jaccard | rule_top_k_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| euclid_q1 | 6075 | 161395 | 0.052 | 0.204 | 0.170 | 0.083 | 0.074 | 0.420 | 0.429 |

## What the numbers say

*Interpret shifts by their most likely physical driver:*

- **Resolution** (Euclid VIS ≈ 0.10"/pix, HST/JWST ≈ 0.03–0.05"/pix vs DECaLS 0.262"/pix): finer sampling sharpens bulge/disk decompositions and moves `sersic_n`, `concentration`, `bulge-size`.
- **Band** (Euclid VIS is broad-optical, JWST is near-IR): changes `smooth-or-featured`, `has-spiral-arms`, `arm-count` because dust and stellar populations dominate the appearance differently at each band.
- **PSF**: distinct PSFs shift `smoothness` and `asymmetry`, which are pixel-scale statistics.

## Rule stability

`rule_mean_jaccard` = mean overlap of features between the reference PySR rule and a fresh PySR fit on the shifted survey, per class. `rule_top_k_overlap` = Jaccard on the top-5 dominant concepts overall. Values near 1 mean the symbolic explanation is portable; values near 0 mean the shifted survey is telling PySR to lean on different concepts entirely.
