# Part 1, RCT analysis

**SRM check**: arm sizes {'control': 21306, 'mens': 21307, 'womens': 21387}, chi-square p = 0.904 → no sample-ratio mismatch.

**Covariate balance**: max |SMD| = 0.0137 (all far below the 0.1 rule of thumb).

## Average treatment effects (vs control)

| treatment | outcome | mean_treat | mean_ctrl | ate | rel_lift_pct | welch_t | p_raw | p_holm | ci_boot_lo | ci_boot_hi | significant_5pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mens | visit | 0.1828 | 0.1062 | 0.0766 | 72.1405 | 22.6202 | 1.36e-112 | 8.19e-112 | 0.0701 | 0.0834 | True |
| mens | conversion | 0.0125 | 0.0057 | 0.0068 | 118.8422 | 7.3897 | 1.50e-13 | 6.01e-13 | 0.0050 | 0.0086 | True |
| mens | spend | 1.4226 | 0.6528 | 0.7698 | 117.9289 | 5.3001 | 1.16e-07 | 3.49e-07 | 0.4840 | 1.0517 | True |
| womens | visit | 0.1514 | 0.1062 | 0.0452 | 42.6055 | 13.9847 | 2.43e-44 | 1.22e-43 | 0.0390 | 0.0516 | True |
| womens | conversion | 0.0088 | 0.0057 | 0.0031 | 54.3313 | 3.7816 | 1.56e-04 | 3.12e-04 | 0.0014 | 0.0048 | True |
| womens | spend | 1.0772 | 0.6528 | 0.4244 | 65.0152 | 3.2564 | 0.0011 | 0.0011 | 0.1519 | 0.6861 | True |


*p_holm = Holm step-down adjustment across the 2 treatments x 3 outcomes family. Bootstrap CIs use 2,000 resamples.*


## Segment-level ATEs on visit

| segment_var | level | treatment | n_treat | n_ctrl | ate_visit | ci_lo | ci_hi | p |
|---|---|---|---|---|---|---|---|---|
| history_segment | 1) $0 - $100 | mens | 7724 | 7612 | 0.0686 | 0.0584 | 0.0787 | 1.75e-39 |
| history_segment | 1) $0 - $100 | womens | 7634 | 7612 | 0.0468 | 0.0369 | 0.0566 | 1.29e-20 |
| history_segment | 2) $100 - $200 | mens | 4691 | 4836 | 0.0691 | 0.0558 | 0.0824 | 2.33e-24 |
| history_segment | 2) $100 - $200 | womens | 4727 | 4836 | 0.0425 | 0.0299 | 0.0551 | 4.59e-11 |
| history_segment | 3) $200 - $350 | mens | 4090 | 4044 | 0.0844 | 0.0682 | 0.1006 | 1.95e-24 |
| history_segment | 3) $200 - $350 | womens | 4155 | 4044 | 0.0361 | 0.0209 | 0.0513 | 3.25e-06 |
| history_segment | 4) $350 - $500 | mens | 2097 | 2124 | 0.0892 | 0.0654 | 0.1130 | 2.25e-13 |
| history_segment | 4) $350 - $500 | womens | 2188 | 2124 | 0.0404 | 0.0180 | 0.0628 | 4.13e-04 |
| history_segment | 5) $500 - $750 | mens | 1597 | 1652 | 0.0990 | 0.0740 | 0.1239 | 1.13e-14 |
| history_segment | 5) $500 - $750 | womens | 1662 | 1652 | 0.0661 | 0.0424 | 0.0898 | 4.92e-08 |
| history_segment | 6) $750 - $1,000 | mens | 644 | 622 | 0.0615 | 0.0188 | 0.1042 | 0.0048 |
| history_segment | 6) $750 - $1,000 | womens | 593 | 622 | 0.0548 | 0.0114 | 0.0982 | 0.0135 |
| history_segment | 7) $1,000 + | mens | 464 | 416 | 0.0971 | 0.0432 | 0.1509 | 4.22e-04 |
| history_segment | 7) $1,000 + | womens | 428 | 416 | 0.0538 | 0.0006 | 0.1070 | 0.0477 |
| channel | Multichannel | mens | 2577 | 2606 | 0.0829 | 0.0626 | 0.1033 | 1.66e-15 |
| channel | Multichannel | womens | 2579 | 2606 | 0.0471 | 0.0276 | 0.0666 | 2.31e-06 |
| channel | Phone | mens | 9240 | 9327 | 0.0756 | 0.0661 | 0.0851 | 5.90e-55 |
| channel | Phone | womens | 9454 | 9327 | 0.0446 | 0.0357 | 0.0535 | 1.01e-22 |
| channel | Web | mens | 9490 | 9373 | 0.0756 | 0.0653 | 0.0859 | 1.60e-46 |
| channel | Web | womens | 9354 | 9373 | 0.0457 | 0.0357 | 0.0556 | 2.96e-19 |
