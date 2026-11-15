"""Exploratory data analysis: 9 figures + data-driven business insights.

Insight sentences are computed from the data at run time (nothing is hard-coded), and are
written to reports/eda_insights.md.  All figures use SYNTHETIC data.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import config  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name):
    d = config.REPORTS_DIR / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(d / f"{name}.png", dpi=110)
    plt.close(fig)


def run_eda(sales, medicines, batches, as_of):
    as_of = pd.Timestamp(as_of)
    m = medicines.set_index("Medicine_ID")
    s = sales.merge(medicines[["Medicine_ID", "Medicine_Name", "Category", "Selling_Price", "Purchase_Price"]],
                    on="Medicine_ID")
    s["Revenue"] = s["Units_Sold"] * s["Selling_Price"]
    s["Profit"] = s["Units_Sold"] * (s["Selling_Price"] - s["Purchase_Price"])
    s["Month"] = s["Date"].dt.to_period("M").dt.to_timestamp()
    ins = {}

    # 1. Top 10 medicines
    top = s.groupby("Medicine_Name")["Units_Sold"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot.barh(ax=ax, color="#2a9d8f")
    ax.set_title("Top 10 most-selling medicines (units, full history)")
    ax.set_xlabel("Units sold")
    _save(fig, "01_top10_medicines")
    share = top.sum() / s["Units_Sold"].sum() * 100
    ins["Top 10 medicines"] = (f"'{top.index[-1]}' is the best seller ({top.iloc[-1]:,.0f} units). The top 10 of "
                               f"{medicines.shape[0]} medicines make up {share:.0f}% of all units - these items "
                               "deserve the tightest stock control.")

    # 2. Monthly demand
    monthly = s.groupby("Month")["Units_Sold"].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653")
    ax.set_title("Monthly total demand (all medicines)")
    ax.set_ylabel("Units sold")
    _save(fig, "02_monthly_demand")
    ins["Monthly demand"] = (f"Demand peaks in {monthly.idxmax():%b %Y} ({monthly.max():,.0f} units) and is lowest in "
                             f"{monthly.idxmin():%b %Y} ({monthly.min():,.0f}); a {monthly.max() / monthly.min():.2f}x "
                             "swing shows purchasing should follow a monthly pattern, not a flat quantity.")

    # 3. Category sales
    cat = s.groupby("Category")["Units_Sold"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    cat.plot.barh(ax=ax, color="#e9c46a")
    ax.set_title("Category-wise sales (units)")
    _save(fig, "03_category_sales")
    ins["Category sales"] = (f"'{cat.index[-1]}' leads with {cat.iloc[-1] / cat.sum() * 100:.0f}% of units; "
                             f"'{cat.index[0]}' is the smallest at {cat.iloc[0] / cat.sum() * 100:.0f}%.")

    # 4. Stock distribution (per medicine, usable stock)
    stock = batches.groupby("Medicine_ID")["Current_Stock"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(stock, bins=20, ax=ax, color="#f4a261")
    ax.set_title("Distribution of total stock per medicine (units)")
    ax.set_xlabel("Units in stock")
    _save(fig, "04_stock_distribution")
    ins["Stock distribution"] = (f"Stock per medicine ranges from {stock.min():,.0f} to {stock.max():,.0f} units "
                                 f"(median {stock.median():,.0f}); the long right tail marks medicines that hold far more "
                                 "stock than typical and should be checked for overstock.")

    # 5. Nearing expiry (units by days-to-expiry bucket)
    b = batches.copy()
    b["Days"] = (b["Expiry_Date"] - as_of).dt.days
    bins = [0, 30, 60, 90, 180, 365, 10_000]
    labels = ["0-30", "31-60", "61-90", "91-180", "181-365", "365+"]
    b["Bucket"] = pd.cut(b["Days"], bins=bins, labels=labels)
    near = b.groupby("Bucket", observed=False)["Current_Stock"].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    near.plot.bar(ax=ax, color=["#e63946", "#e76f51", "#f4a261", "#e9c46a", "#8ab17d", "#2a9d8f"], rot=0)
    ax.set_title("Units in stock by days until expiry (live batches)")
    ax.set_xlabel("Days until expiry")
    _save(fig, "05_near_expiry")
    n90 = near.iloc[:3].sum()
    ins["Nearing expiry"] = (f"{n90:,.0f} units ({n90 / max(near.sum(), 1) * 100:.0f}% of live stock) expire within 90 days "
                             "- these batches need FEFO priority, promotion or supplier return.")

    # 6. Expired inventory
    exp_b = b[b["Days"] <= 0].merge(medicines[["Medicine_ID", "Medicine_Name", "Purchase_Price"]], on="Medicine_ID")
    exp_b["Value"] = exp_b["Current_Stock"] * exp_b["Purchase_Price"]
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(exp_b):
        e = exp_b.sort_values("Value").tail(10)
        ax.barh(e["Medicine_Name"] + " / " + e["Batch_Number"], e["Value"], color="#e63946")
        ax.set_xlabel(f"Value at cost ({config.CURRENCY})")
    else:
        ax.text(0.5, 0.5, "No expired batches in stock", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Expired inventory (value at cost)")
    _save(fig, "06_expired_inventory")
    ins["Expired inventory"] = (f"{len(exp_b)} expired batch(es) are still on the shelf, worth "
                                f"{exp_b['Value'].sum():,.0f} {config.CURRENCY} at cost - already a realised loss that "
                                "must be removed from sale." if len(exp_b) else "No expired stock is on the shelf.")

    # 7. Revenue trend
    rev = s.groupby("Month")[["Revenue", "Profit"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rev.index, rev["Revenue"], marker="o", label="Revenue", color="#264653")
    ax.plot(rev.index, rev["Profit"], marker="o", label="Gross profit", color="#2a9d8f")
    ax.set_title(f"Monthly revenue and gross profit ({config.CURRENCY})")
    ax.legend()
    _save(fig, "07_revenue_trend")
    half = len(rev) // 2
    chg = (rev["Revenue"].iloc[half:].sum() / rev["Revenue"].iloc[:half].sum() - 1) * 100
    ins["Revenue trend"] = (f"Revenue in the second half of the history is {chg:+.1f}% versus the first half; gross margin "
                            f"averages {rev['Profit'].sum() / rev['Revenue'].sum() * 100:.0f}%.")

    # 8. Seasonal demand heat-map (category x month, index = month mean / category mean)
    s["MonthNum"] = s["Date"].dt.month
    pivot = s.groupby(["Category", "MonthNum"])["Units_Sold"].mean().unstack()
    idx = pivot.div(pivot.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(idx, cmap="RdYlGn_r", center=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Seasonal index"})
    ax.set_title("Seasonal demand index (1.0 = category average)")
    ax.set_xlabel("Month")
    _save(fig, "08_seasonal_patterns")
    swing = (idx.max(axis=1) - idx.min(axis=1)).sort_values(ascending=False)
    ins["Seasonal patterns"] = (f"'{swing.index[0]}' is the most seasonal category (index ranges {idx.loc[swing.index[0]].min():.2f}"
                                f"-{idx.loc[swing.index[0]].max():.2f}, peak in month {int(idx.loc[swing.index[0]].idxmax())}); "
                                f"'{swing.index[-1]}' is the steadiest. Seasonal categories need forecast-driven stocking.")

    # 9. Inventory turnover = annualised cost of goods sold / current stock value at cost
    last365 = s[s["Date"] > as_of - pd.Timedelta(days=365)]
    cogs = (last365["Units_Sold"] * last365["Purchase_Price"]).groupby(last365["Category"]).sum()
    bv = batches.merge(medicines[["Medicine_ID", "Category", "Purchase_Price"]], on="Medicine_ID")
    bv = bv[bv["Expiry_Date"] > as_of]
    stock_val = (bv["Current_Stock"] * bv["Purchase_Price"]).groupby(bv["Category"]).sum()
    turn = (cogs / stock_val).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    turn.plot.barh(ax=ax, color="#6a994e")
    ax.set_title("Inventory turnover by category (annual COGS / current stock value)")
    ax.set_xlabel("Turns per year")
    _save(fig, "09_inventory_turnover")
    ins["Inventory turnover"] = (f"'{turn.index[-1]}' turns {turn.iloc[-1]:.1f}x per year (fastest); '{turn.index[0]}' turns "
                                 f"{turn.iloc[0]:.1f}x (slowest) - slow turnover plus a fixed shelf life is where expiry loss concentrates.")

    lines = ["# EDA business insights (auto-generated, SYNTHETIC data)", ""]
    lines += [f"- **{k}** - {v}" for k, v in ins.items()]
    (config.REPORTS_DIR / "eda_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ins
