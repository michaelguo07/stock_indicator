from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stock_indicator_lab import (
    DEFAULT_TICKERS,
    add_features,
    chronological_split,
    fetch_yahoo,
    pct,
    search_custom_indicator,
    signal_metrics,
)


BASE_FEATURES = [
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


@dataclass
class IterationSummary:
    iteration: int
    change: str
    winner: str
    accuracy: float
    buy_accuracy: float
    sell_accuracy: float
    coverage: float
    avg_signal_return: float
    detail: str


def add_research_features(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, df in frame.groupby("ticker", sort=False):
        df = df.sort_values("date").copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"].replace(0, np.nan)

        df["ret_1"] = close.pct_change()
        df["ret_2"] = close.pct_change(2)
        df["ret_3"] = close.pct_change(3)
        df["ema_5_20"] = close.ewm(span=5, adjust=False).mean() / close.ewm(span=20, adjust=False).mean() - 1
        df["ema_20_100"] = close.ewm(span=20, adjust=False).mean() / close.ewm(span=100, adjust=False).mean() - 1
        df["range_pct"] = (high - low) / close
        df["body_pct"] = (close - open_) / open_
        df["upper_wick"] = (high - np.maximum(open_, close)) / close
        df["lower_wick"] = (np.minimum(open_, close) - low) / close
        df["volatility_10"] = close.pct_change().rolling(10).std()
        df["volatility_30"] = close.pct_change().rolling(30).std()
        df["vol_ratio_10_30"] = df["volatility_10"] / df["volatility_30"]
        df["volume_ratio_5_20"] = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
        df["dist_high_20"] = close / high.rolling(20).max() - 1
        df["dist_low_20"] = close / low.rolling(20).min() - 1
        df["drawdown_60"] = close / close.rolling(60).max() - 1
        df["reversal_pressure"] = -df["ret_3"] * df["vol_ratio_10_30"]
        df["trend_quality"] = df["ema_20_100"] / df["volatility_30"]
        parts.append(df)

    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def choose_probability_threshold(
    valid: pd.DataFrame,
    probability: np.ndarray,
    min_signals: int,
    min_coverage: float,
) -> tuple[float, float, float]:
    best = (-1.0, 0.55, 0.45)
    valid_index = valid.index
    for high in np.arange(0.51, 0.66, 0.01):
        for low in np.arange(0.34, 0.50, 0.01):
            signal = pd.Series(np.where(probability >= high, 1, np.where(probability <= low, -1, 0)), index=valid_index)
            active = signal != 0
            if int(active.sum()) < min_signals or float(active.mean()) < min_coverage:
                continue
            expected_up = signal[active] == 1
            actual_up = valid.loc[active, "target"] == 1
            acc = float((expected_up == actual_up).mean())
            avg_return = float((signal[active] * valid.loc[active, "next_return"]).mean())
            score = acc + min(float(active.mean()), 0.25) * 0.05 + max(avg_return, -0.05)
            if score > best[0]:
                best = (score, float(high), float(low))
    return best[1], best[2], best[0]


def evaluate_model(
    name: str,
    model,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    min_signals: int = 30,
    min_coverage: float = 0.005,
):
    model.fit(train[features], train["target"])
    valid_probability = model.predict_proba(valid[features])[:, 1]
    high, low, score = choose_probability_threshold(valid, valid_probability, min_signals, min_coverage)

    combined = pd.concat([train, valid])
    model.fit(combined[features], combined["target"])
    test_probability = model.predict_proba(test[features])[:, 1]
    signal = pd.Series(np.where(test_probability >= high, 1, np.where(test_probability <= low, -1, 0)), index=test.index)
    return signal_metrics(
        test,
        signal,
        name,
        "ai",
        f"features={len(features)}, tuned_buy>={high:.2f}, tuned_sell<={low:.2f}, validation_score={score:.3f}",
    )


def winner_for_iteration(iteration: int, change: str, results) -> IterationSummary:
    ranked = sorted(results, key=lambda item: (item.test_accuracy, item.avg_signal_return, item.coverage), reverse=True)
    winner = ranked[0]
    return IterationSummary(
        iteration=iteration,
        change=change,
        winner=winner.name,
        accuracy=winner.test_accuracy,
        buy_accuracy=winner.buy_accuracy,
        sell_accuracy=winner.sell_accuracy,
        coverage=winner.coverage,
        avg_signal_return=winner.avg_signal_return,
        detail=winner.detail,
    )


def run_iterations() -> list[IterationSummary]:
    frames = [fetch_yahoo(ticker, "2015-01-01", "2026-05-11") for ticker in DEFAULT_TICKERS]
    data = add_research_features(add_features(pd.concat(frames, ignore_index=True), horizon=5))
    train, valid, test = chronological_split(data)

    extra_features = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ema_5_20",
        "ema_20_100",
        "range_pct",
        "body_pct",
        "upper_wick",
        "lower_wick",
        "volatility_10",
        "volatility_30",
        "vol_ratio_10_30",
        "volume_ratio_5_20",
        "dist_high_20",
        "dist_low_20",
        "drawdown_60",
        "reversal_pressure",
        "trend_quality",
    ]
    all_features = BASE_FEATURES + extra_features
    compact_features = [
        "bollinger_b",
        "atr_pct",
        "gap",
        "rsi_14",
        "ema_20_100",
        "vol_ratio_10_30",
        "dist_high_20",
        "dist_low_20",
        "reversal_pressure",
        "trend_quality",
    ]

    iterations = [
        (
            "Baseline logistic model with original indicators",
            [evaluate_model("logistic_base", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), train, valid, test, BASE_FEATURES)],
        ),
        (
            "Tune buy/sell probability thresholds on validation data",
            [evaluate_model("logistic_tuned", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")), train, valid, test, BASE_FEATURES, 20, 0.005)],
        ),
        (
            "Add short-term return, candle, volume, and trend features",
            [evaluate_model("logistic_extra_features", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")), train, valid, test, all_features, 20, 0.005)],
        ),
        (
            "Try stronger regularization to reduce noisy coefficients",
            [evaluate_model("logistic_regularized", make_pipeline(StandardScaler(), LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced")), train, valid, test, all_features, 20, 0.005)],
        ),
        (
            "Try a random forest for nonlinear indicator relationships",
            [
                evaluate_model(
                    "random_forest_tuned",
                    RandomForestClassifier(n_estimators=450, max_depth=6, min_samples_leaf=20, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
                    train,
                    valid,
                    test,
                    all_features,
                    20,
                    0.005,
                )
            ],
        ),
        (
            "Try ExtraTrees for noisier nonlinear searches",
            [
                evaluate_model(
                    "extra_trees_tuned",
                    ExtraTreesClassifier(n_estimators=550, max_depth=5, min_samples_leaf=25, class_weight="balanced", random_state=42, n_jobs=-1),
                    train,
                    valid,
                    test,
                    all_features,
                    20,
                    0.005,
                )
            ],
        ),
        (
            "Try gradient boosting on a compact feature set",
            [
                evaluate_model(
                    "gradient_boost_compact",
                    GradientBoostingClassifier(n_estimators=140, learning_rate=0.035, max_depth=2, min_samples_leaf=35, random_state=42),
                    train,
                    valid,
                    test,
                    compact_features,
                    20,
                    0.005,
                )
            ],
        ),
        (
            "Search a generated composite indicator over expanded features",
            [search_custom_indicator(train, valid, test, all_features, candidates=4000)],
        ),
    ]

    ensemble_models = [
        make_pipeline(StandardScaler(), LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced")),
        ExtraTreesClassifier(n_estimators=550, max_depth=5, min_samples_leaf=25, class_weight="balanced", random_state=42, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=140, learning_rate=0.035, max_depth=2, min_samples_leaf=35, random_state=42),
    ]
    valid_probabilities = []
    test_probabilities = []
    for model, features in zip(ensemble_models, [all_features, all_features, compact_features], strict=True):
        model.fit(train[features], train["target"])
        valid_probabilities.append(model.predict_proba(valid[features])[:, 1])
        model.fit(pd.concat([train, valid])[features], pd.concat([train, valid])["target"])
        test_probabilities.append(model.predict_proba(test[features])[:, 1])
    high, low, score = choose_probability_threshold(valid, np.mean(valid_probabilities, axis=0), 20, 0.005)
    ensemble_signal = pd.Series(np.where(np.mean(test_probabilities, axis=0) >= high, 1, np.where(np.mean(test_probabilities, axis=0) <= low, -1, 0)), index=test.index)
    iterations.append(
        (
            "Blend the best linear, tree, and boosting models",
            [
                signal_metrics(
                    test,
                    ensemble_signal,
                    "ai_probability_ensemble",
                    "ai",
                    f"features=mixed, tuned_buy>={high:.2f}, tuned_sell<={low:.2f}, validation_score={score:.3f}",
                )
            ],
        )
    )

    conservative_results = []
    for min_coverage in [0.01, 0.02, 0.05]:
        conservative_results.append(
            evaluate_model(
                f"logistic_min_coverage_{min_coverage:.0%}",
                make_pipeline(StandardScaler(), LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced")),
                train,
                valid,
                test,
                all_features,
                30,
                min_coverage,
            )
        )
    iterations.append(("Require more coverage so the winner is less fragile", conservative_results))

    return [winner_for_iteration(number, change, results) for number, (change, results) in enumerate(iterations, 1)]


def main() -> None:
    summaries = run_iterations()
    print("\n10-Iteration AI Indicator Improvement Run")
    print("=" * 104)
    print(f"{'iter':<4} {'winner':<26} {'accuracy':>9} {'buy acc':>9} {'sell acc':>9} {'coverage':>9} {'avg ret':>9}  change")
    for item in summaries:
        print(
            f"{item.iteration:<4} {item.winner:<26} {pct(item.accuracy):>9} {pct(item.buy_accuracy):>9} "
            f"{pct(item.sell_accuracy):>9} {pct(item.coverage):>9} {pct(item.avg_signal_return):>9}  {item.change}"
        )

    best = sorted(summaries, key=lambda item: (item.accuracy, item.avg_signal_return, item.coverage), reverse=True)[0]
    print("\nBest final finding")
    print(f"- Iteration {best.iteration}: {best.winner}")
    print(f"- Accuracy: {pct(best.accuracy)}")
    print(f"- Buy accuracy: {pct(best.buy_accuracy)}")
    print(f"- Sell accuracy: {pct(best.sell_accuracy)}")
    print(f"- Coverage: {pct(best.coverage)}")
    print(f"- Average signal return over the next 5 trading days: {pct(best.avg_signal_return)}")
    print(f"- Detail: {best.detail}")


if __name__ == "__main__":
    main()
