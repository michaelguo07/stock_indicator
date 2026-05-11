# AI Stock Indicator Lab

This project searches for stock buy/sell indicators and compares classic indicators against generated custom indicators and a small AI model.

It answers:

- Which indicator had the best out-of-sample accuracy?
- Did a generated/custom indicator beat common indicators?
- What were the buy and sell accuracies?
- What strategy return did each signal produce in the test period?

## How It Works

The script downloads daily OHLCV data from Yahoo Finance's public chart endpoint, builds indicator features, and evaluates them chronologically:

1. Train period: first 70% of rows
2. Validation period: next 15%, used to choose thresholds/formulas
3. Test period: final 15%, used for the reported accuracy

This avoids using future data to score the model.

## Run

```bash
pip install -r requirements.txt
python stock_indicator_lab.py
```

Optional examples:

```bash
python stock_indicator_lab.py --tickers SPY AAPL MSFT NVDA TSLA --start 2018-01-01 --horizon 5
python stock_indicator_lab.py --tickers SPY --custom-candidates 3000
```

## Notes

Accuracy means: when the indicator says buy or sell, how often was the stock direction correct over the next `--horizon` trading days.

This is research code, not financial advice. Markets change, transaction costs matter, and high test accuracy does not guarantee future profits.
