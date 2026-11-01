| method | policy | amazon_base_loss | amazon_base_lock | amazon_base_fail | amazon_base_spend | amazon_entry_loss | amazon_entry_lock | amazon_entry_fail | amazon_entry_spend | domainnet_base_loss | domainnet_base_lock | domainnet_base_fail | domainnet_base_spend | domainnet_entry_loss | domainnet_entry_lock | domainnet_entry_fail | domainnet_entry_spend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Independent | indep | 0.0 | 10.0 |  | 35.0 | 0.0 | 10.0 | 20 | 35.0 | 0.0 | 6.7 |  | 25.0 | 0.0 | 8.1 | 10 | 25.0 |
| Row-level sharing | share_naive | 25.6 | 14.3 |  | 5.7 | 26.7 | 15.5 | 40 | 5.8 | 11.5 | 29.6 |  | 1.7 | 10.3 | 34.2 | 90 | 1.7 |
| QShare without floor | share_agg | 27.9 | 10.5 |  | 5.7 | 27.1 | 9.8 | 25 | 5.8 | 10.9 | 19.6 |  | 1.7 | 14.0 | 21.9 | 45 | 1.7 |
| QShare without floor, with TS | share_agg_ts | 28.2 | 6.5 |  | 4.4 | 28.1 | 6.2 | 15 | 4.5 | 7.7 | 15.4 |  | 1.7 | 11.0 | 15.0 | 35 | 1.7 |
| QShare without floor, with UCB | share_agg_ucb | 29.3 | 8.0 |  | 4.5 | 28.0 | 7.6 | 20 | 4.5 | 9.4 | 18.3 |  | 1.5 | 10.2 | 19.2 | 40 | 1.6 |
| Entry subsidy | fund80_agg |  |  |  |  | 27.3 | 9.5 | 5 | 6.0 |  |  |  |  | 15.1 | 18.8 | 5 | 2.0 |
| QShare | floor80_agg | 28.5 | 3.0 |  | 21.1 | 28.0 | 3.1 | 0 | 21.3 | 5.0 | 0.0 |  | 13.8 | 6.6 | 1.9 | 10 | 14.0 |
