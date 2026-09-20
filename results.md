# Results

| Model | Features | Accuracy | MCC | F1 |
|---|---|---|---|---|
| Majority-class baseline | - | 0.4939 | 0.0000 | 0.6613 |
| BiLSTM + attention | Technical only | 0.5264 | 0.0583 | 0.5807 |
| BiGRU + attention | Technical only | 0.5094 | 0.0406 | 0.6392 |
| BiLSTM + attention | Technical + sentiment | 0.5191 | 0.0440 | 0.5812 |
| BiGRU + attention | Technical + sentiment | 0.5113 | 0.0412 | 0.6318 |

Chronological split at 2015-08-14 | sequence length 5 | 20 epochs | seed 42
