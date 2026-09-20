# Multimodal Stock Movement Prediction via Temporal NLP and Technical Indicators

Predicting next-day directional stock movement (up/down) by fusing technical market
indicators with sentiment extracted from financial micro-blogs, using bidirectional
recurrent networks with a temporal attention mechanism.

Built on the StockNet benchmark dataset (Xu & Cohen, 2018).

---

## Results

87 tickers, 41,766 aligned trading days, January 2014 – December 2015.
Chronological split at 2015-08-14: 33,327 training rows, 8,439 validation rows.
Sequence length 5, 20 epochs, seed 42. Reported epoch selected on validation MCC.

| Model | Features | Accuracy | MCC | F1 |
|---|---|---|---|---|
| Majority-class baseline | — | 0.4939 | 0.0000 | 0.6613 |
| BiLSTM + attention | Technical only | **0.5264** | **0.0583** | 0.5807 |
| BiGRU + attention | Technical only | 0.5094 | 0.0406 | 0.6392 |
| BiLSTM + attention | Technical + sentiment | 0.5191 | 0.0440 | 0.5812 |
| BiGRU + attention | Technical + sentiment | 0.5113 | 0.0412 | 0.6318 |

**What the numbers show.** The best configuration reaches MCC 0.058 — above zero, but
only just. Every model sits within roughly three percentage points of a coin flip.

Three things are worth reading carefully rather than skimming:

**Every model's F1 is below the baseline's.** The majority-class baseline predicts "up"
for every single day, which gives it perfect recall and an F1 of 0.661 — higher than any
trained model here. A reader who ranked these rows by F1 would conclude the best model is
the one that does no work at all. This is precisely why MCC is the metric this project
reports: it is exactly 0.000 for that degenerate baseline, and it is the only column in
the table that separates a model with a weak signal from a model with none.

**Sentiment features did not help.** Adding the seven tweet-derived features to the
BiLSTM moved MCC from 0.0583 down to 0.0440. The same pattern appeared in earlier work on
this dataset. Daily lexicon-based sentiment on retail micro-blogs appears to add noise
faster than it adds signal, at least at this granularity.

**The models peak almost immediately.** Three of the four configurations recorded their
best validation MCC at epoch 2 of 20. Training loss continued to fall from roughly 0.693
to 0.665 across the remaining epochs while validation MCC did not improve. That is
memorisation of the training window, not a signal that generalises forward in time.

Taken together, this is the result the literature would predict. Next-day direction from
public price history and public tweets is close to unpredictable, which is what the
weak-form efficient market hypothesis says it should be. A project of this shape finding
a large edge would be more likely to have a leak than a discovery.

---

## Approach

**Target.** Binary next-day direction: `1` if the next close exceeds today's close,
else `0`. The validation set is 49.4% up-days, so the task is near-balanced.

**Technical features (23).** Returns and log returns; SMA over 5/10/20 days; EMA 12/26;
MACD and signal line; RSI(14); Bollinger bands with width and position; momentum over
5/10 days; rolling volatility over 5/10 days; volume change, volume SMA and volume
ratio; high–low and close–open ratios.

**Sentiment features (7).** Per-ticker, per-day aggregates from the tweet corpus:
tweet count, TextBlob polarity and subjectivity means, and VADER positive, negative,
neutral and compound means.

**Calendar alignment.** Price data exists only on trading days; tweets exist every day.
The two streams are joined with a backward `merge_asof`, which attaches to each trading
day the most recent tweet features at or before that date. Weekend and holiday chatter
therefore rolls forward into the next active session, and no future information can
reach a prediction.

**Chronological split.** Training and validation are separated by calendar date, with
every ticker cut at the same day. A random row-level split would let the model see a
stock's Thursday while predicting its Wednesday — on panel data that leak is easy to
introduce and hard to see in the metrics.

**Sequence construction.** Sliding windows of `sequence_length` trading days (default 5)
predict the direction of the following day. Windows are built one ticker at a time. The
aligned dataset is a vertical stack of per-ticker frames, so windowing it as a single
array would produce sequences that straddle two companies and carry a label belonging to
neither.

**Scaling.** `StandardScaler` is fit on the training rows only and applied to both
splits, so no validation-period statistics reach the fitted transform.

**Models.** A shared architecture with a switchable recurrent core:

- Bidirectional LSTM or GRU, hidden size 64, 2 layers
- Additive temporal attention over the recurrent outputs, producing a weighted context
  vector rather than relying on the final hidden state
- Classifier head: Linear(128 → 32) → ReLU → Dropout(0.3) → Linear(32 → 1) → Sigmoid

**Training.** Adam, learning rate 1e-3, weight decay 1e-5, binary cross-entropy,
20 epochs. Accuracy, F1 and MCC are computed on the validation set each epoch, and the
reported epoch is the one with the highest MCC.

---

## Repository structure

```
stocknet-prediction/
├── src/
│   ├── data_preprocessing.py   # Technical indicators + tweet sentiment extraction
│   ├── data_merger.py          # Calendar alignment and multimodal fusion
│   ├── dataset.py              # PyTorch sliding-window sequence dataset
│   ├── models.py               # BiLSTM / BiGRU with temporal attention
│   ├── train.py                # Training loop and evaluation
│   └── run_experiments.py      # Entry point: split, scale, train, report
├── requirements.txt
└── README.md
```

---

## Running it

```bash
git clone https://github.com/Dgupta3321/stocknet-prediction.git
cd stocknet-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -c "import nltk; nltk.download('vader_lexicon')"
```

Download the StockNet dataset from
[yumoxu/stocknet-dataset](https://github.com/yumoxu/stocknet-dataset) and place it as:

```
data/
├── price/raw/              # per-ticker CSV price history
└── tweet/preprocessed/     # per-ticker daily tweet JSON
```

Then run everything:

```bash
python src/run_experiments.py
```

The first run builds the aligned dataset and caches it to `data/aligned_dataset.csv`;
afterwards use `--skip-build` to go straight to training. Results are written to
`results.md`.

```bash
python src/run_experiments.py --skip-build --epochs 30 --seed 1
```

---

## Notes and limitations

- **Model selection uses the validation set.** The reported epoch is chosen by validation
  MCC and the same split is then reported as the result, which makes these numbers
  optimistic. A held-out test period, untouched during selection, would give a more
  honest estimate. With effects this small, that distinction is not academic.

- **Single seed, single split.** Every number is one run at seed 42 with one cutoff date.
  At an MCC of 0.058 the run-to-run spread could plausibly cover the gap between the
  configurations, so the ranking between models should not be read as established.

- **Non-finite values in volume-derived features.** `Volume_Change`, `Volume_Ratio` and
  `High_Low_Ratio` divide by quantities that can be zero, producing infinities rather
  than NaNs — which means a plain `dropna()` does not catch them. Affected rows are
  converted and dropped before scaling.

- **Zero-filled sentiment is not neutral.** Tickers and days with no tweet coverage get
  sentiment features of zero, which the model cannot distinguish from genuinely neutral
  sentiment. A missingness indicator would be the better treatment.

- **Lexicon sentiment, not finance-tuned.** VADER and TextBlob are general-purpose. A
  domain model such as FinBERT reads financial language considerably better — "beat
  expectations" and "shorting this" are not general-English sentiment.

- **One market regime.** 2014–2015 is a single, broadly rising window. Nothing here
  demonstrates that the approach transfers to a different volatility environment.

- **Attention weights are not inspected.** The attention layer produces per-timestep
  weights that are currently discarded. Examining which days in the 5-day window the
  model leans on would be the most informative next step.

---

## Reference

Xu, Y. and Cohen, S.B. (2018). *Stock Movement Prediction from Tweets and Historical
Prices.* Proceedings of the 56th Annual Meeting of the Association for Computational
Linguistics.
