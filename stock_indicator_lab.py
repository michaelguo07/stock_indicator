from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "XOM"]
RNG_SEED = 42


@dataclass
class IndicatorResult:
    name: str
    kind: str
    test_accuracy: float
    buy_accuracy: float
    sell_accuracy: float
    coverage: float
    avg_signal_return: float
    buy_signals: int
    sell_signals: int
    detail: str


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
        f"?period1={int(start_dt.timestamp())}&period2={int(end_dt.timestamp())}"
        "&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"No data returned for {ticker}: {error}")

    item = result[0]
    quote = item["indicators"]["quote"][0]
    adjclose = item["indicators"].get("adjclose", [{}])[0].get("adjclose")
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(item["timestamp"], unit="s").date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": adjclose if adjclose is not None else quote["close"],
            "volume": quote["volume"],
        }
    )
    raw["date"] = pd.to_datetime(raw["date"])
    raw["ticker"] = ticker.upper()
    return raw.dropna().sort_values("date")[["ticker", "date", "open", "high", "low", "close", "volume"]]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, df in frame.groupby("ticker", sort=False):
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].replace(0, np.nan)

        ret_1 = close.pct_change()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        lowest_14 = low.rolling(14).min()
        highest_14 = high.rolling(14).max()
        obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()

        df["sma_10_50"] = (sma_10 / sma_50) - 1
        df["sma_20_50"] = (sma_20 / sma_50) - 1
        df["rsi_14"] = (rsi(close, 14) - 50) / 50
        df["macd_hist"] = (macd - macd_signal) / close
        df["bollinger_b"] = (close - (mid - 2 * std)) / (4 * std)
        df["stoch_14"] = ((close - lowest_14) / (highest_14 - lowest_14)) - 0.5
        df["atr_pct"] = atr / close
        df["momentum_5"] = close.pct_change(5)
        df["momentum_20"] = close.pct_change(20)
        df["vol_z_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        df["obv_trend"] = obv.pct_change(10).replace([np.inf, -np.inf], np.nan)
        df["gap"] = (df["open"] / close.shift()) - 1
        df["next_return"] = close.shift(-horizon) / close - 1
        df["target"] = (df["next_return"] > 0).astype(int)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    first = int(len(ordered) * 0.70)
    second = int(len(ordered) * 0.85)
    return ordered.iloc[:first], ordered.iloc[first:second], ordered.iloc[second:]


def signal_metrics(df: pd.DataFrame, signal: pd.Series, name: str, kind: str, detail: str) -> IndicatorResult:
    signal = signal.reindex(df.index).fillna(0).astype(int)
    active = signal != 0
    if active.sum() == 0:
        return IndicatorResult(name, kind, 0, 0, 0, 0, 0, 0, 0, detail)

    expected_up = signal[active] == 1
    actual_up = df.loc[active, "target"] == 1
    correct = expected_up == actual_up
    buy = signal == 1
    sell = signal == -1
    buy_accuracy = float(((df.loc[buy, "target"] == 1).mean()) if buy.any() else 0)
    sell_accuracy = float(((df.loc[sell, "target"] == 0).mean()) if sell.any() else 0)
    signed_returns = signal[active] * df.loc[active, "next_return"]
    avg_signal_return = float(signed_returns.mean())
    return IndicatorResult(
        name=name,
        kind=kind,
        test_accuracy=float(correct.mean()),
        buy_accuracy=buy_accuracy,
        sell_accuracy=sell_accuracy,
        coverage=float(active.mean()),
        avg_signal_return=avg_signal_return,
        buy_signals=int(buy.sum()),
        sell_signals=int(sell.sum()),
        detail=detail,
    )


def choose_threshold(train: pd.DataFrame, valid: pd.DataFrame, feature: str) -> tuple[float, int, float]:
    values = train[feature].dropna()
    quantiles = np.linspace(0.1, 0.9, 17)
    thresholds = sorted(set(float(values.quantile(q)) for q in quantiles if np.isfinite(values.quantile(q))))
    best = (-1.0, 0.0, 1)
    for threshold in thresholds:
        for direction in (1, -1):
            raw = valid[feature] * direction
            signal = pd.Series(np.where(raw > abs(threshold), 1, np.where(raw < -abs(threshold), -1, 0)), index=valid.index)
            active = signal != 0
            if active.sum() < 30:
                continue
            pred = (signal[active] == 1).astype(int)
            score = balanced_accuracy_score(valid.loc[active, "target"], pred)
            if score > best[0]:
                best = (float(score), abs(threshold), direction)
    return best[1], best[2], best[0]


def evaluate_classic_indicators(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> list[IndicatorResult]:
    results = []
    for feature in features:
        threshold, direction, score = choose_threshold(train, valid, feature)
        raw = test[feature] * direction
        signal = pd.Series(np.where(raw > threshold, 1, np.where(raw < -threshold, -1, 0)), index=test.index)
        results.append(
            signal_metrics(
                test,
                signal,
                feature,
                "classic",
                f"threshold={threshold:.5f}, direction={direction}, validation_bal_acc={score:.3f}",
            )
        )
    return results


def zscore_from_train(train: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    means = train[features].mean()
    stds = train[features].std().replace(0, 1)
    return (frame[features] - means) / stds


def search_custom_indicator(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, features: list[str], candidates: int) -> IndicatorResult:
    rng = np.random.default_rng(RNG_SEED)
    z_train = zscore_from_train(train, train, features)
    z_valid = zscore_from_train(train, valid, features)
    z_test = zscore_from_train(train, test, features)
    best = (-1.0, None)

    for _ in range(candidates):
        selected = rng.choice(features, size=3, replace=False)
        weights = rng.normal(0, 1, size=3)
        interaction_weight = rng.normal(0, 0.35)
        valid_score = (
            z_valid[selected[0]] * weights[0]
            + z_valid[selected[1]] * weights[1]
            + z_valid[selected[2]] * weights[2]
            + interaction_weight * z_valid[selected[0]] * z_valid[selected[1]]
        )
        train_score = (
            z_train[selected[0]] * weights[0]
            + z_train[selected[1]] * weights[1]
            + z_train[selected[2]] * weights[2]
            + interaction_weight * z_train[selected[0]] * z_train[selected[1]]
        )
        temp_train = train.copy()
        temp_train["custom_score"] = train_score
        temp_valid = valid.copy()
        temp_valid["custom_score"] = valid_score
        threshold, direction, score = choose_threshold(temp_train, temp_valid, "custom_score")
        if score > best[0]:
            best = (score, (selected, weights, interaction_weight, threshold, direction))

    selected, weights, interaction_weight, threshold, direction = best[1]
    test_score = (
        z_test[selected[0]] * weights[0]
        + z_test[selected[1]] * weights[1]
        + z_test[selected[2]] * weights[2]
        + interaction_weight * z_test[selected[0]] * z_test[selected[1]]
    )
    raw = test_score * direction
    signal = pd.Series(np.where(raw > threshold, 1, np.where(raw < -threshold, -1, 0)), index=test.index)
    detail = (
        f"{weights[0]:+.2f}*z({selected[0]}) {weights[1]:+.2f}*z({selected[1]}) "
        f"{weights[2]:+.2f}*z({selected[2]}) {interaction_weight:+.2f}*z({selected[0]})*z({selected[1]}), "
        f"threshold={threshold:.3f}, direction={direction}, validation_bal_acc={best[0]:.3f}"
    )
    return signal_metrics(test, signal, "generated_composite", "custom", detail)


def evaluate_ai_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> list[IndicatorResult]:
    models = {
        "ai_logistic_composite": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "ai_random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=25,
            max_depth=5,
            class_weight="balanced_subsample",
            random_state=RNG_SEED,
            n_jobs=-1,
        ),
    }
    results = []
    for name, model in models.items():
        model.fit(train[features], train["target"])
        proba = model.predict_proba(test[features])[:, 1]
        signal = pd.Series(np.where(proba >= 0.55, 1, np.where(proba <= 0.45, -1, 0)), index=test.index)
        results.append(signal_metrics(test, signal, name, "ai", "buy when p(up)>=0.55, sell when p(up)<=0.45"))
    return results


def pct(value: float) -> str:
    if value == 0 or not math.isfinite(value):
        return "0.0%"
    return f"{value * 100:.1f}%"


def print_report(results: list[IndicatorResult], test: pd.DataFrame, horizon: int) -> None:
    ranked = sorted(results, key=lambda r: (r.test_accuracy, r.avg_signal_return), reverse=True)
    baseline = float(test["target"].mean())
    print("\nAI Stock Indicator Lab")
    print("=" * 72)
    print(f"Test rows: {len(test):,} | Horizon: {horizon} trading days | Baseline up-rate: {pct(baseline)}")
    print("\nBest indicators by out-of-sample signal accuracy:")
    print(
        f"{'rank':<4} {'indicator':<24} {'type':<8} {'accuracy':>9} {'buy acc':>9} "
        f"{'sell acc':>9} {'coverage':>9} {'avg ret':>9}"
    )
    for i, item in enumerate(ranked[:12], 1):
        print(
            f"{i:<4} {item.name:<24} {item.kind:<8} {pct(item.test_accuracy):>9} "
            f"{pct(item.buy_accuracy):>9} {pct(item.sell_accuracy):>9} "
            f"{pct(item.coverage):>9} {pct(item.avg_signal_return):>9}"
        )

    winner = ranked[0]
    print("\nWinner")
    print(f"- {winner.name} ({winner.kind})")
    print(f"- Test accuracy: {pct(winner.test_accuracy)}")
    print(f"- Buy accuracy: {pct(winner.buy_accuracy)} across {winner.buy_signals} buy signals")
    print(f"- Sell accuracy: {pct(winner.sell_accuracy)} across {winner.sell_signals} sell signals")
    print(f"- Detail: {winner.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and compare stock buy/sell indicators.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--custom-candidates", type=int, default=1500)
    args = parser.parse_args()

    frames = []
    for ticker in args.tickers:
        try:
            frames.append(fetch_yahoo(ticker, args.start, args.end))
            print(f"Downloaded {ticker}")
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")
    if not frames:
        raise SystemExit("No market data downloaded. Check tickers or network access.")

    data = add_features(pd.concat(frames, ignore_index=True), args.horizon)
    train, valid, test = chronological_split(data)
    features = [
        "sma_10_50",
        "sma_20_50",
        "rsi_14",
        "macd_hist",
        "bollinger_b",
        "stoch_14",
        "atr_pct",
        "momentum_5",
        "momentum_20",
        "vol_z_20",
        "obv_trend",
        "gap",
    ]

    results = []
    results.extend(evaluate_classic_indicators(train, valid, test, features))
    results.append(search_custom_indicator(train, valid, test, features, args.custom_candidates))
    results.extend(evaluate_ai_models(pd.concat([train, valid]), test, features))
    print_report(results, test, args.horizon)


if __name__ == "__main__":
    main()
