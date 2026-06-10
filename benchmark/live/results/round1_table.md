| Task | Tokens before | Tokens after | Saved | Saved % | TAS Δ | Pass held |
|---|---:|---:|---:|---:|---:|:---:|
| csv-filter | 5538 | 5505 | 33 | 0.6% | -1.9 | ✅ |
| csv-filter.r2 | 4968 | 5446 | -478 | -9.6% | -5.0 | ✅ |
| dedupe-helpers | 5781 | 5032 | 749 | 13.0% | +0.7 | ✅ |
| dedupe-helpers.r2 | 4627 | 5262 | -635 | -13.7% | -0.3 | ✅ |
| fix-imports | 3280 | 3939 | -659 | -20.1% | -2.8 | ✅ |
| fix-imports.r2 | 4566 | 4533 | 33 | 0.7% | +0.6 | ✅ |
| fix-offby-one | 3793 | 4123 | -330 | -8.7% | -2.9 | ✅ |
| fix-offby-one.r2 | 3801 | 3476 | 325 | 8.6% | +1.1 | ✅ |
| implement-median | 4861 | 5154 | -293 | -6.0% | -4.8 | ✅ |
| implement-median.r2 | 3648 | 3764 | -116 | -3.2% | +0.7 | ✅ |
| rename-api | 4223 | 5211 | -988 | -23.4% | -2.1 | ✅ |
| rename-api.r2 | 5432 | 5704 | -272 | -5.0% | -1.1 | ✅ |

**Aggregate over 12 task(s):** mean token reduction **-5.6%** (95% bootstrap CI [-11.4%, 0.2%]); mean TAS delta -1.5 (95% CI [-2.7, -0.4]).

**Pass rate:** 12/12 before → 12/12 after — constant task outcome on every pair (the savings are at unchanged pass rate).

**Estimate accuracy** (measured / audit-estimated savings, 12 task(s) with fixes JSON): mean -102% (95% CI [-205%, 0%]).
