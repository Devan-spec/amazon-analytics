"""
ElevanceSkills Amazon Product Analytics Project (optimized)
Same six tasks and same output files as the original, with faster implementations.

Run:
    pip install -r requirements.txt
    python amazon_product_analytics.py --input amazon.csv --output outputs
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.feature_selection import f_regression
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
DPI = 110  # lower than 160: much faster savefig, still crisp

# Compiled once instead of re-compiled for every row.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
JUNK_RE = re.compile(r"[^A-Za-z0-9\s.,!?']")
WS_RE = re.compile(r"\s+")
NUM_JUNK_RE = re.compile(r"[^\d.\-]")


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def clean_numeric(series, remove_percent=False):
    s = series.astype(str)
    if remove_percent:
        s = s.str.replace("%", "", regex=False)
    # one regex pass removes commas, spaces, currency symbols, etc.
    return pd.to_numeric(s.str.replace(NUM_JUNK_RE, "", regex=True), errors="coerce")


def clean_text_series(series):
    """Vectorized text cleaning (replaces row-wise .apply)."""
    return (series.fillna("").astype(str)
            .str.replace(URL_RE, " ", regex=True)
            .str.replace(TAG_RE, " ", regex=True)
            .str.replace(JUNK_RE, " ", regex=True)
            .str.replace(WS_RE, " ", regex=True)
            .str.strip())


def join_unique_text(raw, col):
    """Unique non-empty texts per product joined by space (no per-group lambda)."""
    t = raw.loc[raw[col].str.len() > 0, ["product_id", col]].drop_duplicates()
    return t.groupby("product_id")[col].agg(" ".join)


def prepare_data(input_path, out_dir):
    raw = pd.read_csv(input_path, low_memory=False)
    required = {
        "product_id", "product_name", "category", "discounted_price",
        "actual_price", "discount_percentage", "rating", "rating_count",
        "review_title", "review_content"
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    raw["discounted_price"] = clean_numeric(raw["discounted_price"])
    raw["actual_price"] = clean_numeric(raw["actual_price"])
    raw["discount_percentage"] = clean_numeric(raw["discount_percentage"], True)
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")
    raw["rating_count"] = clean_numeric(raw["rating_count"])
    raw["main_category"] = (
        raw["category"].fillna("Unknown").astype(str).str.split("|").str[0]
    )
    raw["review_title_clean"] = clean_text_series(raw["review_title"])
    raw["review_content_clean"] = clean_text_series(raw["review_content"])

    # Fast aggregation: built-in reducers only, no Python lambdas per group.
    data = raw.groupby("product_id").agg(
        product_name=("product_name", "first"),
        discounted_price=("discounted_price", "median"),
        actual_price=("actual_price", "median"),
        discount_percentage=("discount_percentage", "median"),
        rating=("rating", "median"),
        rating_count=("rating_count", "median"),
    )
    # Mode of category (ties -> alphabetical, same as Series.mode().iloc[0]).
    cat_mode = (raw.groupby(["product_id", "main_category"]).size()
                .reset_index(name="n")
                .sort_values(["product_id", "n", "main_category"],
                             ascending=[True, False, True])
                .drop_duplicates("product_id")
                .set_index("product_id")["main_category"])
    data["main_category"] = cat_mode
    data["review_title_clean"] = join_unique_text(raw, "review_title_clean")
    data["review_content_clean"] = join_unique_text(raw, "review_content_clean")
    data[["review_title_clean", "review_content_clean"]] = (
        data[["review_title_clean", "review_content_clean"]].fillna("")
    )
    data = data.reset_index()

    n_unique = len(data)
    data = data[
        data["discount_percentage"].between(0, 100)
        & data["rating"].between(1, 5)
        & (data["discounted_price"] > 0)
        & (data["actual_price"] > 0)
        & (data["rating_count"] >= 0)
    ].copy()

    data["log_rating_count"] = np.log1p(data["rating_count"])
    data["log_actual_price"] = np.log1p(data["actual_price"])
    data["log_discounted_price"] = np.log1p(data["discounted_price"])

    data.to_csv(out_dir / "clean_product_level_data.csv", index=False)
    pd.DataFrame({
        "metric": ["raw_rows", "unique_product_ids", "product_level_rows",
                   "rows_removed_by_validation"],
        "value": [len(raw), n_unique, len(data), n_unique - len(data)]
    }).to_csv(out_dir / "data_quality_summary.csv", index=False)
    return len(raw), data


# Text-heavy columns are excluded from derived exports (smaller, faster writes).
TEXT_COLS = ["review_title_clean", "review_content_clean"]


def task1_segmentation(data, out):
    cols = ["discounted_price", "discount_percentage", "rating", "log_rating_count"]
    X = data[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X)

    # Fit each k once and keep the labels; the original refit the best model again.
    scores, labels_by_k = [], {}
    for k in range(2, min(9, len(data) - 1)):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3)
        labels = km.fit_predict(X_scaled)
        labels_by_k[k] = labels
        scores.append({"k": k, "inertia": km.inertia_,
                       "silhouette": silhouette_score(X_scaled, labels)})
    score_df = pd.DataFrame(scores)
    score_df.to_csv(out / "task1_cluster_selection.csv", index=False)
    best_k = int(score_df.loc[score_df["silhouette"].idxmax(), "k"])

    data["kmeans_segment"] = labels_by_k[best_k]
    km_sil = score_df.loc[score_df["k"] == best_k, "silhouette"].iloc[0]

    alt_labels = AgglomerativeClustering(
        n_clusters=best_k, linkage="ward").fit_predict(X_scaled)
    comparison = pd.DataFrame([
        {"method": "KMeans", "clusters": best_k, "silhouette": km_sil},
        {"method": "Agglomerative", "clusters": best_k,
         "silhouette": silhouette_score(X_scaled, alt_labels)},
    ])
    comparison.to_csv(out / "task1_method_comparison.csv", index=False)

    profile = data.groupby("kmeans_segment").agg(
        products=("product_id", "count"),
        avg_discounted_price=("discounted_price", "mean"),
        avg_discount_percentage=("discount_percentage", "mean"),
        avg_rating=("rating", "mean"),
        median_rating_count=("rating_count", "median")
    ).reset_index()
    profile.to_csv(out / "task1_segment_profiles.csv", index=False)

    for col, title, name in [("inertia", "Elbow Method", "elbow"),
                             ("silhouette", "Silhouette Score by k", "silhouette")]:
        plt.figure(figsize=(8, 4))
        plt.plot(score_df["k"], score_df[col], marker="o")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia" if col == "inertia" else "Silhouette score")
        plt.title(title)
        save_plot(out / f"task1_{name}.png")

    data.drop(columns=TEXT_COLS).to_csv(out / "data_with_segments.csv", index=False)
    return profile, comparison


def task2_discount_rating(data, out):
    d = data.dropna(subset=[
        "discount_percentage", "rating", "actual_price",
        "discounted_price", "rating_count", "main_category"
    ]).copy()

    plt.figure(figsize=(8, 5))
    sns.regplot(data=d, x="discount_percentage", y="rating",
                scatter_kws={"alpha": 0.35, "s": 20}, line_kws={"color": "red"})
    plt.title("Discount Percentage vs Rating")
    save_plot(out / "task2_discount_rating_scatter.png")

    bins = [-0.001, 20, 40, 60, 80, 100]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    d["discount_range"] = pd.cut(d["discount_percentage"], bins=bins,
                                 labels=labels, include_lowest=True)
    band = d.groupby("discount_range", observed=False).agg(
        product_count=("product_id", "count"),
        average_rating=("rating", "mean"),
        median_rating=("rating", "median")
    ).reset_index()
    band.to_csv(out / "task2_rating_by_discount_band.csv", index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(band["discount_range"].astype(str), band["average_rating"])
    plt.xlabel("Discount range")
    plt.ylabel("Average rating")
    plt.title("Average Rating by Discount Range")
    save_plot(out / "task2_discount_band_ratings.png")

    p_corr, p_p = pearsonr(d["discount_percentage"], d["rating"])
    s_corr, s_p = spearmanr(d["discount_percentage"], d["rating"])
    pd.DataFrame([
        {"method": "Pearson", "correlation": p_corr, "p_value": p_p},
        {"method": "Spearman", "correlation": s_corr, "p_value": s_p}
    ]).to_csv(out / "task2_correlations.csv", index=False)

    groups = [g.values for _, g in d.groupby("discount_range", observed=True)["rating"]
              if len(g) >= 2]
    anova_stat, anova_p = f_oneway(*groups) if len(groups) >= 2 else (np.nan, np.nan)
    pd.DataFrame([{"test": "One-way ANOVA", "statistic": anova_stat,
                   "p_value": anova_p}]).to_csv(
        out / "task2_discount_band_anova.csv", index=False)

    # log columns already exist in `data`; no recomputation needed.
    model = smf.ols("rating ~ discount_percentage + log_actual_price + "
                    "log_rating_count + C(main_category)", data=d).fit(cov_type="HC3")
    (out / "task2_regression_summary.txt").write_text(model.summary().as_text())

    # Category correlations: one groupby pass, size filter up-front.
    category_rows = []
    for category, g in d.groupby("main_category"):
        if len(g) >= 10 and g["discount_percentage"].nunique() > 1:
            r, p = pearsonr(g["discount_percentage"], g["rating"])
            category_rows.append({"category": category, "n": len(g),
                                  "pearson_r": r, "p_value": p})
    cat_corr = pd.DataFrame(category_rows)
    cat_corr.to_csv(out / "task2_category_correlations.csv", index=False)
    if not cat_corr.empty:
        plt.figure(figsize=(10, 5))
        ordered = cat_corr.sort_values("pearson_r")
        plt.barh(ordered["category"], ordered["pearson_r"])
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Discount-Rating Correlation by Main Category")
        save_plot(out / "task2_category_correlations.png")

    interaction = smf.ols(
        "rating ~ discount_percentage * C(main_category) + "
        "log_actual_price + log_rating_count", data=d).fit(cov_type="HC3")
    (out / "task2_interaction_regression_summary.txt").write_text(
        interaction.summary().as_text())

    filtered = d
    for col in ["discount_percentage", "actual_price", "rating_count"]:
        q1, q3 = filtered[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        filtered = filtered[filtered[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    pd.DataFrame([
        {"sample": "Original", "n": len(d), "pearson_r": p_corr},  # reuse, not recomputed
        {"sample": "IQR filtered sensitivity sample", "n": len(filtered),
         "pearson_r": pearsonr(filtered["discount_percentage"], filtered["rating"])[0]}
    ]).to_csv(out / "task2_outlier_sensitivity.csv", index=False)


def task3_sentiment(data, out):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError as exc:
        raise ImportError("Install vaderSentiment: pip install vaderSentiment") from exc

    text = (data["review_title_clean"] + ". " + data["review_content_clean"]).str.strip()
    # Score each distinct text once; list comprehension avoids Series.apply overhead.
    uniq = text.unique()
    score_map = dict(zip(uniq, (analyzer.polarity_scores(t)["compound"] for t in uniq)))
    compound = text.map(score_map)
    data["sentiment_score"] = compound
    data["sentiment"] = np.select([compound >= 0.05, compound <= -0.05],
                                  ["positive", "negative"], default="neutral")

    data["sentiment"].value_counts().rename_axis("sentiment").reset_index(
        name="products").to_csv(out / "task3_sentiment_distribution.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x="sentiment", order=["positive", "neutral", "negative"])
    plt.title("Overall Review Sentiment")
    save_plot(out / "task3_sentiment_distribution.png")

    # One crosstab feeds the proportions, chi-square test, and negative share.
    contingency = pd.crosstab(data["main_category"], data["sentiment"])
    props = contingency.div(contingency.sum(axis=1), axis=0)
    props.stack().rename("proportion").reset_index().to_csv(
        out / "task3_category_sentiment.csv", index=False)

    plt.figure(figsize=(11, 6))
    sns.countplot(data=data, y="main_category", hue="sentiment",
                  order=data["main_category"].value_counts().index)
    plt.title("Sentiment by Main Category")
    save_plot(out / "task3_category_sentiment.png")

    chi2, chi_p, dof, _ = chi2_contingency(contingency)
    pd.DataFrame([{"test": "Chi-square", "chi2": chi2,
                   "degrees_freedom": dof, "p_value": chi_p}]).to_csv(
        out / "task3_category_sentiment_test.csv", index=False)

    neg = props["negative"] if "negative" in props else pd.Series(0.0, index=props.index)
    neg.sort_values(ascending=False).rename("negative_share").reset_index().to_csv(
        out / "task3_negative_category_share.csv", index=False)

    mismatch = (((data["rating"] >= 4.0) & (data["sentiment"] == "negative")) |
                ((data["rating"] <= 3.0) & (data["sentiment"] == "positive")))
    data["rating_sentiment_mismatch"] = mismatch
    data.loc[mismatch, ["product_id", "product_name", "rating", "sentiment",
                        "sentiment_score"] + TEXT_COLS].to_csv(
        out / "task3_rating_sentiment_mismatches.csv", index=False)
    data[["product_id", "sentiment", "sentiment_score"]].to_csv(
        out / "task3_product_sentiment.csv", index=False)


def minmax(a):
    a = np.asarray(a, dtype=float)
    rng = a.max(axis=0) - a.min(axis=0)
    rng = np.where(rng == 0, 1.0, rng)
    return (a - a.min(axis=0)) / rng


def task4_ranking(data, out):
    d = data.drop(columns=TEXT_COLS).copy()
    C = d["rating"].mean()
    m = d["rating_count"].quantile(0.60)
    rc = d["rating_count"]
    d["weighted_rating"] = (rc / (rc + m)) * d["rating"] + (m / (rc + m)) * C

    norm = minmax(np.column_stack([
        d["weighted_rating"], np.log1p(rc),
        d["discount_percentage"], np.log1p(d["discounted_price"])
    ]))
    d["rating_norm"], d["popularity_norm"], d["discount_norm"], d["price_norm"] = norm.T
    d["affordability_norm"] = 1 - d["price_norm"]

    M = d[["rating_norm", "popularity_norm", "discount_norm",
           "affordability_norm"]].to_numpy()
    w1 = np.array([0.45, 0.25, 0.15, 0.15])
    w2 = np.array([0.40, 0.30, 0.15, 0.15])
    d["ranking_score"] = M @ w1       # one matrix-vector product per weight set
    d["alternative_score"] = M @ w2

    d.nlargest(10, "ranking_score").to_csv(out / "task4_top10_products.csv", index=False)
    top_a = set(d.nlargest(10, "ranking_score")["product_id"])
    top_b = set(d.nlargest(10, "alternative_score")["product_id"])
    pd.DataFrame([{
        "top10_overlap_count": len(top_a & top_b),
        "top10_overlap_fraction": len(top_a & top_b) / 10,
        "original_weights": str(dict(zip(["rating_norm", "popularity_norm",
                                          "discount_norm", "affordability_norm"], w1))),
        "alternative_weights": str(dict(zip(["rating_norm", "popularity_norm",
                                             "discount_norm", "affordability_norm"], w2))),
    }]).to_csv(out / "task4_weight_sensitivity.csv", index=False)
    d.to_csv(out / "task4_all_product_scores.csv", index=False)


def task5_anomalies(data, out):
    d = data.drop(columns=TEXT_COLS).copy()
    features = ["actual_price", "discounted_price", "discount_percentage",
                "rating", "log_rating_count"]
    X = d[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Vectorized IQR over all columns at once (no per-row apply).
    q = X.quantile([0.25, 0.75])
    iqr = q.loc[0.75] - q.loc[0.25]
    flags = (X < q.loc[0.25] - 1.5 * iqr) | (X > q.loc[0.75] + 1.5 * iqr)
    names = np.array(features)
    d["iqr_anomaly"] = flags.any(axis=1)
    d["iqr_anomaly_fields"] = [", ".join(names[r]) for r in flags.to_numpy()]

    iso = IsolationForest(n_estimators=150, contamination="auto",
                          random_state=RANDOM_STATE, n_jobs=-1)
    d["isolation_forest_anomaly"] = iso.fit_predict(
        StandardScaler().fit_transform(X)) == -1

    d["anomaly_group"] = np.select(
        [d["iqr_anomaly"] & d["isolation_forest_anomaly"],
         d["iqr_anomaly"], d["isolation_forest_anomaly"]],
        ["Both methods", "IQR only", "Isolation Forest only"],
        default="Not flagged")
    d.to_csv(out / "task5_all_anomaly_flags.csv", index=False)
    d[d["iqr_anomaly"] | d["isolation_forest_anomaly"]].sort_values(
        ["iqr_anomaly", "isolation_forest_anomaly"], ascending=False
    ).head(50).to_csv(out / "task5_top_anomalies_to_investigate.csv", index=False)

    # Individual boxplot files are kept for compatibility, but drawn cheaply.
    for col in ["actual_price", "discounted_price", "discount_percentage",
                "rating", "rating_count"]:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=d[col])
        plt.title(f"Boxplot: {col}")
        save_plot(out / f"task5_boxplot_{col}.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=d, x="actual_price", y="discounted_price",
                    hue="anomaly_group", alpha=0.8)
    plt.title("Actual vs Discounted Price: Anomaly Flags")
    save_plot(out / "task5_price_anomaly_scatter.png")


def task6_rating_prediction(data, out):
    d = data
    numeric = ["actual_price", "discounted_price", "discount_percentage", "rating_count"]
    categorical = ["main_category"]
    X = d[numeric + categorical]
    y = d["rating"]

    def make_pre():
        return ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             categorical)
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)

    # RF n_jobs=1 because cross_validate already parallelizes over folds.
    # (Original used n_jobs=-1 in both -> thread oversubscription.)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=1)
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", make_pre()), ("model", estimator)])
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv, n_jobs=-1,
            scoring={"mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error", "r2": "r2"})
        pipe.fit(X_train, y_train)
        if name == "RandomForest":
            pipe.named_steps["model"].set_params(n_jobs=-1)  # fast predict/importances
        pred = pipe.predict(X_test)
        results.append({
            "model": name,
            "test_MAE": mean_absolute_error(y_test, pred),
            "test_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "test_R2": r2_score(y_test, pred),
            "CV_MAE_mean": -cv_scores["test_mae"].mean(),
            "CV_RMSE_mean": -cv_scores["test_rmse"].mean(),
            "CV_R2_mean": cv_scores["test_r2"].mean()
        })
        fitted[name] = (pipe, pred)

    results_df = pd.DataFrame(results).sort_values("test_MAE")
    results_df.to_csv(out / "task6_model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipe, best_pred = fitted[best_name]
    pd.DataFrame({
        "actual_rating": y_test.values, "predicted_rating": best_pred,
        "absolute_error": np.abs(y_test.values - best_pred)
    }).to_csv(out / "task6_best_model_test_predictions.csv", index=False)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=best_pred, alpha=0.7)
    plt.plot([1, 5], [1, 5], linestyle="--", color="red")
    plt.xlabel("Actual rating")
    plt.ylabel("Predicted rating")
    plt.title(f"Actual vs Predicted Rating ({best_name})")
    save_plot(out / "task6_actual_vs_predicted.png")

    # Reuse the already-fitted preprocessor instead of fitting it a third time.
    pre = fitted["LinearRegression"][0].named_steps["preprocess"]
    Xt = pre.transform(X_train)
    f_scores, f_p = f_regression(Xt, y_train)
    pd.DataFrame({"feature": pre.get_feature_names_out(),
                  "f_score": f_scores, "p_value": f_p}).sort_values(
        "f_score", ascending=False).to_csv(
        out / "task6_feature_selection.csv", index=False)

    rf_pipe = fitted["RandomForest"][0]
    fi = pd.DataFrame({
        "feature": rf_pipe.named_steps["preprocess"].get_feature_names_out(),
        "importance": rf_pipe.named_steps["model"].feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(
        out / "task6_random_forest_feature_importance.csv", index=False)
    plt.figure(figsize=(9, 6))
    top_fi = fi.nlargest(15, "importance").sort_values("importance")
    plt.barh(top_fi["feature"], top_fi["importance"])
    plt.title("Random Forest: Top Feature Importances")
    save_plot(out / "task6_feature_importance.png")

    joblib.dump(best_pipe, out / "best_rating_prediction_model.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="amazon.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing dataset...")
    n_raw, data = prepare_data(args.input, out)
    print(f"Raw rows: {n_raw:,}; clean products: {len(data):,}")

    print("\nTask 1: Product segmentation")
    profile, comparison = task1_segmentation(data, out)
    print(profile.to_string(index=False))
    print(comparison.to_string(index=False))

    print("\nTask 2: Discount-rating relationship")
    task2_discount_rating(data, out)

    print("\nTask 3: Review sentiment analysis")
    task3_sentiment(data, out)

    print("\nTask 4: Product ranking")
    task4_ranking(data, out)

    print("\nTask 5: Anomaly detection")
    task5_anomalies(data, out)

    print("\nTask 6: Predictive rating analysis")
    task6_rating_prediction(data, out)

    print(f"\nCompleted. All output files are in: {out.resolve()}")


if __name__ == "__main__":
    main()
