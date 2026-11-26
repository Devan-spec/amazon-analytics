"""
ElevanceSkills Amazon Product Analytics — Streamlit Dashboard
Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

OUT = Path("outputs")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Analytics",
    page_icon="📦",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📦 Amazon Product Analytics Dashboard")
st.caption("ElevanceSkills Internship Project · 6-Task Analysis on Amazon India Product Data")
st.divider()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
sections = [
    "🏠 Overview",
    "1️⃣  Product Segmentation",
    "2️⃣  Discount vs Rating",
    "3️⃣  Sentiment Analysis",
    "4️⃣  Product Ranking",
    "5️⃣  Anomaly Detection",
    "6️⃣  Rating Prediction",
]
choice = st.sidebar.radio("Go to section", sections)
st.sidebar.divider()
st.sidebar.markdown("**Dataset:** Amazon India products  \n**Products analysed:** 1,348")

# ── Helper ─────────────────────────────────────────────────────────────────────
def load(name):
    p = OUT / name
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def show_image(name, caption="", width=None):
    p = OUT / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=(width is None))

def metric_row(cols_values: dict):
    cols = st.columns(len(cols_values))
    for col, (label, val) in zip(cols, cols_values.items()):
        col.metric(label, val)

# ══════════════════════════════════════════════════════════════════════════════
# 0. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if choice == "🏠 Overview":
    st.header("Project Overview")

    dq = load("data_quality_summary.csv")
    if not dq.empty:
        d = dq.set_index("metric")["value"]
        metric_row({
            "Raw Rows":            f"{int(d.get('raw_rows', 0)):,}",
            "Unique Product IDs":  f"{int(d.get('unique_product_ids', 0)):,}",
            "Clean Products":      f"{int(d.get('product_level_rows', 0)):,}",
            "Rows Dropped":        f"{int(d.get('rows_removed_by_validation', 0)):,}",
        })

    st.divider()
    st.subheader("What this project covers")
    tasks = {
        "1 — Product Segmentation":   "KMeans + Agglomerative clustering to group products by price, discount, rating and popularity.",
        "2 — Discount vs Rating":     "Pearson/Spearman correlations, ANOVA across discount bands, OLS regression with category fixed effects.",
        "3 — Sentiment Analysis":     "VADER sentiment on all review text; category-level breakdown and rating–sentiment mismatch detection.",
        "4 — Product Ranking":        "Bayesian weighted score combining rating, popularity, discount and affordability.",
        "5 — Anomaly Detection":      "IQR method + Isolation Forest to surface pricing and rating outliers.",
        "6 — Rating Prediction":      "Linear Regression vs Random Forest; 5-fold CV, feature importance and model persistence.",
    }
    for title, desc in tasks.items():
        st.markdown(f"**{title}** — {desc}")

    st.divider()
    st.subheader("Clean dataset preview")
    clean = load("clean_product_level_data.csv")
    if not clean.empty:
        display_cols = ["product_id", "product_name", "main_category",
                        "discounted_price", "actual_price",
                        "discount_percentage", "rating", "rating_count"]
        st.dataframe(clean[display_cols].head(50), use_container_width=True)
        st.caption(f"{len(clean):,} products total — showing first 50")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "1️⃣  Product Segmentation":
    st.header("Task 1 — Product Segmentation")

    st.subheader("Cluster selection")
    sel = load("task1_cluster_selection.csv")
    if not sel.empty:
        c1, c2 = st.columns(2)
        with c1:
            show_image("task1_elbow.png", "Elbow Method")
        with c2:
            show_image("task1_silhouette.png", "Silhouette Score by k")
        best_k = int(sel.loc[sel["silhouette"].idxmax(), "k"])
        best_sil = sel["silhouette"].max()
        st.info(f"Best k = **{best_k}** (silhouette = {best_sil:.4f})")
        st.dataframe(sel, use_container_width=True)

    st.divider()
    st.subheader("Method comparison")
    comp = load("task1_method_comparison.csv")
    if not comp.empty:
        st.dataframe(comp, use_container_width=True)

    st.divider()
    st.subheader("Segment profiles")
    prof = load("task1_segment_profiles.csv")
    if not prof.empty:
        st.dataframe(prof.style.format({
            "avg_discounted_price":    "₹{:.0f}",
            "avg_discount_percentage": "{:.1f}%",
            "avg_rating":              "{:.2f}",
            "median_rating_count":     "{:,.0f}",
        }), use_container_width=True)

        st.divider()
        st.subheader("Segment interpretation")
        for _, row in prof.iterrows():
            seg = int(row["kmeans_segment"])
            price = row["avg_discounted_price"]
            disc  = row["avg_discount_percentage"]
            rat   = row["avg_rating"]
            cnt   = row["median_rating_count"]

            if price > 10000:
                label = "🏷️ Premium"
            elif disc > 50:
                label = "🔥 Budget / Heavy-Discount"
            else:
                label = "⭐ Mid-range / Popular"

            st.markdown(
                f"**Segment {seg} — {label}**  \n"
                f"Products: {int(row['products'])} · "
                f"Avg price: ₹{price:,.0f} · "
                f"Avg discount: {disc:.1f}% · "
                f"Avg rating: {rat:.2f} · "
                f"Median reviews: {cnt:,.0f}"
            )

# ══════════════════════════════════════════════════════════════════════════════
# 2. DISCOUNT vs RATING
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "2️⃣  Discount vs Rating":
    st.header("Task 2 — Discount vs Rating Relationship")

    st.subheader("Scatter plot")
    show_image("task2_discount_rating_scatter.png")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation coefficients")
        corr = load("task2_correlations.csv")
        if not corr.empty:
            st.dataframe(corr.style.format({"correlation": "{:.4f}", "p_value": "{:.2e}"}),
                         use_container_width=True)
            r = corr.loc[corr["method"] == "Pearson", "correlation"].values[0]
            p = corr.loc[corr["method"] == "Pearson", "p_value"].values[0]
            st.info(f"Pearson r = **{r:.4f}**, p = {p:.2e}  \n"
                    "Weak but statistically significant negative relationship — "
                    "higher discounts are associated with marginally lower ratings.")

    with c2:
        st.subheader("ANOVA across discount bands")
        anova = load("task2_discount_band_anova.csv")
        if not anova.empty:
            st.dataframe(anova.style.format({"statistic": "{:.3f}", "p_value": "{:.2e}"}),
                         use_container_width=True)

    st.divider()
    st.subheader("Average rating by discount band")
    band = load("task2_rating_by_discount_band.csv")
    if not band.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            show_image("task2_discount_band_ratings.png")
        with c2:
            st.dataframe(band.style.format({
                "average_rating": "{:.3f}",
                "median_rating":  "{:.2f}",
            }), use_container_width=True)

    st.divider()
    st.subheader("Category-level correlations")
    cat = load("task2_category_correlations.csv")
    if not cat.empty:
        show_image("task2_category_correlations.png")
        st.dataframe(cat.style.format({"pearson_r": "{:.4f}", "p_value": "{:.4f}"}),
                     use_container_width=True)

    st.divider()
    st.subheader("Outlier sensitivity check")
    sens = load("task2_outlier_sensitivity.csv")
    if not sens.empty:
        st.dataframe(sens.style.format({"pearson_r": "{:.4f}"}), use_container_width=True)
        st.caption("Correlation direction and magnitude are stable after IQR filtering.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "3️⃣  Sentiment Analysis":
    st.header("Task 3 — Review Sentiment Analysis")

    dist = load("task3_sentiment_distribution.csv")
    if not dist.empty:
        d = dist.set_index("sentiment")["products"]
        total = d.sum()
        metric_row({
            "✅ Positive": f"{int(d.get('positive', 0)):,}  ({d.get('positive', 0)/total*100:.1f}%)",
            "⚠️ Negative": f"{int(d.get('negative', 0)):,}  ({d.get('negative', 0)/total*100:.1f}%)",
            "😐 Neutral":  f"{int(d.get('neutral',  0)):,}  ({d.get('neutral',  0)/total*100:.1f}%)",
        })

    c1, c2 = st.columns(2)
    with c1:
        show_image("task3_sentiment_distribution.png", "Overall sentiment")
    with c2:
        show_image("task3_category_sentiment.png", "Sentiment by category")

    st.divider()
    st.subheader("Negative sentiment share by category")
    neg = load("task3_negative_category_share.csv")
    if not neg.empty:
        st.dataframe(neg.style.format({"negative_share": "{:.2%}"}),
                     use_container_width=True)

    st.divider()
    st.subheader("Chi-square independence test")
    chi = load("task3_category_sentiment_test.csv")
    if not chi.empty:
        st.dataframe(chi.style.format({"chi2": "{:.3f}", "p_value": "{:.4f}"}),
                     use_container_width=True)
        p = chi["p_value"].values[0]
        if p > 0.05:
            st.info(f"p = {p:.4f} — sentiment distribution does **not** differ "
                    "significantly across categories.")
        else:
            st.warning(f"p = {p:.4f} — sentiment distribution differs "
                       "significantly across categories.")

    st.divider()
    st.subheader("Rating–Sentiment mismatches")
    mis = load("task3_rating_sentiment_mismatches.csv")
    if not mis.empty:
        st.caption(f"{len(mis)} products where high-rated products have negative sentiment "
                   "or low-rated products have positive sentiment.")
        st.dataframe(mis[["product_name", "rating", "sentiment",
                           "sentiment_score"]].head(30),
                     use_container_width=True)
    else:
        st.info("No mismatches found.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RANKING
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "4️⃣  Product Ranking":
    st.header("Task 4 — Product Ranking")

    st.subheader("Top 10 products by composite score")
    st.caption(
        "Score = 0.45 × Bayesian rating + 0.25 × log-popularity + "
        "0.15 × discount + 0.15 × affordability (all min-max normalised)"
    )
    top = load("task4_top10_products.csv")
    if not top.empty:
        show_cols = ["product_name", "main_category", "rating",
                     "rating_count", "discount_percentage",
                     "discounted_price", "ranking_score"]
        st.dataframe(
            top[show_cols].style.format({
                "rating":               "{:.1f}",
                "rating_count":         "{:,.0f}",
                "discount_percentage":  "{:.0f}%",
                "discounted_price":     "₹{:,.0f}",
                "ranking_score":        "{:.4f}",
            }),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Weight sensitivity analysis")
    ws = load("task4_weight_sensitivity.csv")
    if not ws.empty:
        overlap = int(ws["top10_overlap_count"].values[0])
        st.metric("Top-10 overlap when weights shift", f"{overlap}/10 products")
        st.dataframe(ws[["top10_overlap_count", "top10_overlap_fraction"]],
                     use_container_width=True)
        st.caption("9/10 products remain in the top 10 when popularity weight "
                   "is raised from 0.25 → 0.30, confirming ranking stability.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "5️⃣  Anomaly Detection":
    st.header("Task 5 — Anomaly / Outlier Detection")

    all_flags = load("task5_all_anomaly_flags.csv")
    if not all_flags.empty:
        counts = all_flags["anomaly_group"].value_counts()
        total  = len(all_flags)
        metric_row({
            "Total products":          f"{total:,}",
            "Not flagged":             f"{counts.get('Not flagged', 0):,}",
            "Both methods":            f"{counts.get('Both methods', 0):,}",
            "IQR only":                f"{counts.get('IQR only', 0):,}",
            "Isolation Forest only":   f"{counts.get('Isolation Forest only', 0):,}",
        })

    st.divider()
    show_image("task5_price_anomaly_scatter.png", "Actual vs Discounted Price — Anomaly Groups")

    st.divider()
    st.subheader("Boxplots — numeric feature distributions")
    cols_imgs = [
        ("actual_price",        "Actual Price"),
        ("discounted_price",    "Discounted Price"),
        ("discount_percentage", "Discount %"),
        ("rating",              "Rating"),
        ("rating_count",        "Rating Count"),
    ]
    row1, row2, row3 = st.columns(3), st.columns(3), st.columns(2)
    grid = [row1[0], row1[1], row1[2], row2[0], row2[1]]
    for cell, (col, caption) in zip(grid, cols_imgs):
        with cell:
            show_image(f"task5_boxplot_{col}.png", caption)

    st.divider()
    st.subheader("Top anomalies to investigate")
    top_a = load("task5_top_anomalies_to_investigate.csv")
    if not top_a.empty:
        show_cols = ["product_name", "actual_price", "discounted_price",
                     "discount_percentage", "rating", "rating_count",
                     "anomaly_group", "iqr_anomaly_fields"]
        st.dataframe(
            top_a[show_cols].head(50).style.format({
                "actual_price":         "₹{:,.0f}",
                "discounted_price":     "₹{:,.0f}",
                "discount_percentage":  "{:.0f}%",
                "rating":               "{:.1f}",
                "rating_count":         "{:,.0f}",
            }),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# 6. RATING PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif choice == "6️⃣  Rating Prediction":
    st.header("Task 6 — Predictive Rating Analysis")

    st.subheader("Model comparison")
    mc = load("task6_model_comparison.csv")
    if not mc.empty:
        best = mc.iloc[0]["model"]
        st.dataframe(
            mc.style.format({
                "test_MAE":    "{:.4f}",
                "test_RMSE":   "{:.4f}",
                "test_R2":     "{:.4f}",
                "CV_MAE_mean": "{:.4f}",
                "CV_RMSE_mean":"{:.4f}",
                "CV_R2_mean":  "{:.4f}",
            }).highlight_min(subset=["test_MAE"], color="#d4edda"),
            use_container_width=True,
        )
        st.info(
            f"Best model: **{best}**  \n"
            "R² is intentionally low (~0.07–0.14) — price, discount and category "
            "alone explain only ~7–14% of rating variance. Ratings are driven by "
            "product quality and experience factors not present in this dataset."
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_image("task6_actual_vs_predicted.png", "Actual vs Predicted Rating")
    with c2:
        show_image("task6_feature_importance.png", "Random Forest Feature Importance")

    st.divider()
    st.subheader("Feature importance (Random Forest)")
    fi = load("task6_random_forest_feature_importance.csv")
    if not fi.empty:
        st.dataframe(
            fi.style.format({"importance": "{:.4f}"}),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Test set predictions (best model)")
    preds = load("task6_best_model_test_predictions.csv")
    if not preds.empty:
        st.dataframe(
            preds.style.format({
                "actual_rating":    "{:.2f}",
                "predicted_rating": "{:.2f}",
                "absolute_error":   "{:.3f}",
            }),
            use_container_width=True,
        )
        mae = preds["absolute_error"].mean()
        st.metric("Mean Absolute Error on test set", f"{mae:.4f} stars")
