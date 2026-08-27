# Recommendation Report

| Issue | Severity | Confidence | Expected ML Impact | Recommended Action | Risk | Gain |
|-------|----------|------------|--------------------|--------------------|------|------|
| Corrupted files | High | 100% | Breaks Dataloader | Remove from dataset / regenerate | Low | High |
| Train/Test Leakage | Critical | 100% | Inflated validation metrics | Remove leaky instances from Train | Low | Critical |
| Conflicting labels | High | 100% | Poor convergence | Manually review `mislabel_candidates.csv` and fix labels | Medium | High |
| Exact Duplicates (within Train) | Medium | 100% | Data redundancy, biased metrics | Drop duplicates keeping first | Low | Medium |
| Outliers (dims/aspect ratio) | Low | Medium | Might cause resizing artifacts | Resize carefully, pad to square if aspect ratio is extreme | Low | Low |

*Note: All recommended candidates are saved in their respective CSV files.*
