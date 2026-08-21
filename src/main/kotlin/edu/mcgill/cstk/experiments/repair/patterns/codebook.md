| Pattern | Evidence | Any-match repairs | % !dfaRecognized | Primary repairs | Observed rules | Automaton complexity |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Affine gap run (open=1, extension=0) | rule-free structural | 462 | 71.2% | 422 | - | Theta(nd), constant gap modes |
| Bounded 1-to-many rule candidate | rule-inventory candidate | 333 | 51.3% | 31 | 215 | O(nd rho B) after fixing rho merge/split rules |
| Wrapper-pair rule candidate | rule-inventory candidate | 266 | 41.0% | 77 | 104 | Theta(nd) after fixing the wrapper-pair inventory |
| Tandem block duplication or contraction | rule-free structural | 34 | 5.2% | 34 | - | O(ndB) macro arcs |
| Bounded adjacent-block swap | rule-free structural | 11 | 1.7% | 8 | - | O(ndB^2) macro arcs |
| One arbitrary non-adjacent symbol swap | rule-free structural | 6 | 0.9% | 3 | - | O(nq^2), O(n^2) on a fixed alphabet |
| Bounded block reversal | rule-free structural | 3 | 0.5% | 3 | - | O(ndB) macro arcs |
| Adjacent transposition | rule-free structural | 1 | 0.2% | 1 | - | Theta(nd) |
| Token-symbol repetition or unduplication | rule-free structural | 1 | 0.2% | 1 | - | Theta(nd), fixed maximum repetition |