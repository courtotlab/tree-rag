# Frozen results

## Public MultiHop-RAG

The study uses a balanced 200-question sample: 50 comparison, 50 inference, 50 temporal,
and 50 null questions. Official retrieval is evaluated on the 150 non-null questions.

| System | QA accuracy (n=200) | Hits@4 | Hits@10 | MAP@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| TreeQuest evaluated-v0 | 0.495 | 0.6667 | 0.6933 | 0.2258 | 0.4612 |
| Flat hybrid | 0.415 | unavailable | unavailable | unavailable | unavailable |
| Collapsed-tree control | 0.325 | 0.3533 | 0.4800 | 0.1058 | 0.2539 |
| Oracle diagnostic | 0.435 | 1.0000 | 1.0000 | 0.6700 | 1.0000 |

TreeQuest's retrieval advantage over collapsed search is large across all four official
retrieval metrics. Official QA also favors TreeQuest over flat hybrid by 8.0 points and
collapsed search by 17.0 points. The official QA rule is any token intersection with the
gold answer; the oracle's lower QA score than TreeQuest shows why it is a coarse
compatibility measure rather than a semantic gold standard.

A separate joint `gpt-oss:120b` judge scores TreeQuest 0.639 and flat hybrid 0.598. The
paired difference is 0.041, with 35 wins, 138 ties, 27 losses, bootstrap 95% CI
[-0.02925, 0.11150], and two-sided sign-flip `p=0.26177`. This direction is encouraging
but statistically uncertain and is not presented as a confirmed public semantic win.

TreeQuest averages 598.7 seconds, 127.9 model calls, and 4.37 evidence pieces per public
question. The hierarchy build takes 49,401 seconds (13 h 42 min) and 19,976 model calls,
producing a 28.7 MB JSON tree with 20,495 nodes and 19,212 chunks.

## Restricted deployment aggregate

| Metric | TreeQuest evaluated-v0 | Deployed hybrid |
|---|---:|---:|
| Held-out questions | 294 | 294 |
| Mean judged quality | 0.5629 | 0.4686 |
| Mean wall-clock seconds | 586.7 | 84.4 |

Paired difference: 0.0943; bootstrap 95% CI [0.0628, 0.1265]; Wilcoxon
`p=7.24e-8`; paired Cohen's `d_z=0.338`; wins/ties/losses 59/230/5. These are
approved aggregate statistics only. The underlying corpus and per-question records are
not released.

On a later frozen 63-question rerun, TreeQuest scored 0.4965 and the hybrid scored
0.5024 (paired difference -0.0059, 95% CI [-0.0957, 0.0822], sign-flip `p=0.902`).
This small-sample non-replication is reported rather than hidden and motivates the public
study, human validation, and conservative generalization claims.

## Evaluated-v0 ablations (n=63)

| Arm | Quality | Difference vs thorough | 95% CI | Mean seconds |
|---|---:|---:|---:|---:|
| Thorough 120B | 0.4965 | - | - | 427 |
| Contrast off | 0.4441 | -0.0524 | [-0.1275, 0.0219] | 285 |
| Quick budgets | 0.3416 | -0.1549 | [-0.2376, -0.0779] | 169 |
| Shape directives off | 0.4830 | -0.0135 | [-0.1097, 0.0803] | 410 |
| 20B sensitivity arm | 0.4700 | -0.0265 | [-0.1125, 0.0592] | 540 |

Only the quick-budget effect is resolved at this sample size. Contrast is a plausible
quality-cost tradeoff, while directive and model-size differences remain inconclusive.
Every headline and control result uses 120B; 20B is an ablation only.
