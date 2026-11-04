# Eval results

Last run: 2026-11-04

Setup: embeddings hashing, hybrid search, exact quotes, no LLM

| Group | Correct |
| --- | --- |
| direct | 10/11 |
| table | 7/8 |
| conflict | 1/1 |
| paraphrase | 0/3 |
| unanswerable | 11/11 |
| heldout | 6/7 |

- Total correct: 35/41
- False answers: 0
- Right section in top 6: 29/29
- Mean reciprocal rank: 0.98
- Answered by: {'extractive': 22}
- Refusal reasons: {'no_exact_match': 11, 'weak_retrieval': 6, 'policy_is_silent': 1, 'outside_policy_scope': 1}
