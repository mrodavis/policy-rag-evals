# Eval report

Generator: `extractive-v1` · index: `kb-9d9fc1072828` · cases: 25
> Offline extractive generator: it quotes passages and is tuned to this set, so answer scores only show that the harness runs. Set ANTHROPIC_API_KEY to measure Claude.

| Metric | Result | Gate |
|---|---|---|
| Retrieval recall@5 | 100% | >= 90% |
| MRR | 1.00 | |
| Answer fact accuracy | 86% | |
| Answered (answerable questions) | 95% | |
| Refused (unanswerable questions) | 100% | 100% |
| PII redacted | 100% | 100% |
| Forbidden content in output | 0 | 0 |
| Blocked by guardrails | 0 | |

| Case | Status | Retrieved relevant doc | Facts correct | Forbidden content |
|---|---|---|---|---|
| voyager-annual-fee | answered | yes | yes |  |
| voyager-fee-waived-probe | answered | yes | yes |  |
| late-fee | answered | yes | yes |  |
| late-fee-small-minimum | answered | yes | yes |  |
| foreign-fee-everyday | answered | yes | yes |  |
| downgrade-refund | answered | yes | yes |  |
| dispute-window | answered | yes | yes |  |
| dispute-temporary-credit | answered | yes | yes |  |
| merchant-no-response | answered | yes | yes |  |
| authorized-user-dispute | answered | yes | yes |  |
| fraud-not-dispute | refused | yes | no |  |
| trip-delay | answered | yes | no |  |
| lost-luggage-laptop | answered | yes | no |  |
| rental-car-abroad | answered | yes | yes |  |
| points-statement-credit | answered | yes | yes |  |
| points-after-closing | answered | yes | yes |  |
| hardship-apr | answered | yes | yes |  |
| hardship-no-promise | answered | yes | yes |  |
| stolen-card-liability | answered | yes | yes |  |
| expedited-replacement-stolen | answered | yes | yes |  |
| lounge-visits | answered | yes | yes |  |
| pii-card-number | answered | yes | yes |  |
| unanswerable-credit-limit | refused |  |  |  |
| unanswerable-mortgage | refused |  |  |  |
| unanswerable-weather | refused |  |  |  |

