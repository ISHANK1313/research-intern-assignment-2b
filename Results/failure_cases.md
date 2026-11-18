# Fusion model failure cases

Most confident misclassifications of the multimodal model on the shared test split (n=420).

| # | Case type | File | True | Predicted (fusion) | Conf. | P(true) | Speech-only | Text-only |
|---|---|---|---|---|---|---|---|---|
| 1 | fusion error (test split) | `OAF_lease_happy.wav` | happiness | pleasant_surprise | 0.870 | 0.129 | pleasant_surprise (wrong) | pleasant_surprise |
| 2 | fusion near-miss (correct, lowest margin) | `YAF_hash_disgust.wav` | disgust | disgust | 0.706 | 0.706 | pleasant_surprise (wrong) | pleasant_surprise |
| 3 | fusion near-miss (correct, lowest margin) | `OAF_rush_disgust.wav` | disgust | disgust | 0.815 | 0.815 | disgust (ok) | pleasant_surprise |
| 4 | fusion near-miss (correct, lowest margin) | `OAF_wire_ps.wav` | pleasant_surprise | pleasant_surprise | 0.882 | 0.882 | pleasant_surprise (ok) | pleasant_surprise |
| 5 | fusion near-miss (correct, lowest margin) | `OAF_burn_ps.wav` | pleasant_surprise | pleasant_surprise | 0.914 | 0.914 | pleasant_surprise (ok) | pleasant_surprise |
