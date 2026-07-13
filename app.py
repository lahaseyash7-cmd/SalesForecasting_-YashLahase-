import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Page Configuration

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)
st.markdown("""
# Sales Forecasting & Demand Intelligence Dashboard

### Business Analytics using Python & Machine Learning

---
""")

# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"],dayfirst=True)

    if "Shipping Days" not in df.columns:
        df["Shipping Days"] = (
            df["Ship Date"] - df["Order Date"]
        ).dt.days
    return df

df = load_data()

# Sidebar Filters

st.sidebar.title("Dashboard Filters")

st.sidebar.markdown(
"""
Use the filters below to explore sales across
different regions and product categories.
"""
)

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")

forecast_horizon = st.sidebar.slider(
    "Select Forecast Horizon (Months)",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

st.sidebar.write(f"Forecast Period: **{forecast_horizon} Month(s)**")

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]

# KPI Cards

total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique()
total_products = filtered_df["Product Name"].nunique()
total_customers = filtered_df["Customer ID"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.info(f"### Total Sales\n## ${total_sales:,.0f}")
with col2:
    st.success(f"### Orders\n## {total_orders}")
with col3:
    st.warning(f"### Products\n## {total_products}")
with col4:
    st.error(f"### Customers\n## {total_customers}")

st.divider()

# Dataset Preview

st.subheader("Dataset Preview")
st.dataframe(
    filtered_df.head(10),
    use_container_width=True,
    hide_index=True
)

st.info(f"📈 Sales Forecast Horizon Selected: **{forecast_horizon} Month(s)**")

# Monthly Sales Trend

st.subheader("Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
    .sum()
)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    linewidth=3
)
ax.fill_between(
    monthly_sales.index,
    monthly_sales.values,
    alpha=0.3
)
ax.grid(alpha=0.3)
ax.set_title(
    "Monthly Sales Trend",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.grid(True)

st.pyplot(fig)

# Sales Forecast

st.subheader("📈 Sales Forecast")

# Monthly sales
monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
    .sum()
    .reset_index()
)

# Create numeric month index
monthly_sales["Month_Index"] = np.arange(len(monthly_sales))

# Train Linear Regression
X = monthly_sales[["Month_Index"]]
y = monthly_sales["Sales"]

model = LinearRegression()
model.fit(X, y)

# Predict future months
future_index = np.arange(
    len(monthly_sales),
    len(monthly_sales) + forecast_horizon
).reshape(-1, 1)

future_sales = model.predict(future_index)

future_dates = pd.date_range(
    start=monthly_sales["Order Date"].max() + pd.DateOffset(months=1),
    periods=forecast_horizon,
    freq="ME"
)

# Plot
fig, ax = plt.subplots(figsize=(10,5))

# Historical Sales
ax.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o",
    linewidth=2,
    label="Historical Sales"
)

# Forecast Sales
ax.plot(
    future_dates,
    future_sales,
    marker="o",
    linestyle="--",
    linewidth=2,
    color="red",
    label="Forecast"
)

ax.set_title("Sales Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# Sales by Category

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales by Category")
    category_sales = filtered_df.groupby("Category")["Sales"].sum()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(
    x=category_sales.index,
    y=category_sales.values,
    ax=ax
    )
    plt.xticks(rotation=20)
    ax.set_ylabel("Sales")
    st.pyplot(fig)

# Sales by Region

with col2:
    st.subheader("Sales by Region")
    region_sales = filtered_df.groupby("Region")["Sales"].sum()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(
    x=region_sales.index,
    y=region_sales.values,
    ax=ax
    )
    ax.set_ylabel("Sales")
    st.pyplot(fig)

# Top Products

st.subheader("Top 10 Products by Sales")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(
    x=top_products.values,
    y=top_products.index,
    ax=ax
)

ax.set_xlabel("Sales")

st.pyplot(fig)

# Shipping Days

st.subheader("Shipping Days by Region")

shipping = filtered_df.groupby("Region")["Shipping Days"].mean()

fig, ax = plt.subplots(figsize=(7,4))
sns.barplot(
    x=shipping.index,
    y=shipping.values,
    ax=ax
)
ax.set_title("Average Shipping Days")
ax.set_ylabel("Average Days")
st.pyplot(fig)

# Cluster Visualization

if "Cluster" in filtered_df.columns and "PCA1" in filtered_df.columns:
    st.subheader("Product Clusters")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        data=filtered_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="Set2",
        ax=ax
    )
    st.pyplot(fig)

# Anomaly Detection

if "Anomaly" in filtered_df.columns:
    st.subheader("Sales Anomalies")
    fig, ax = plt.subplots(figsize=(10,5))
    normal = filtered_df[filtered_df["Anomaly"] == 1]
    anomaly = filtered_df[filtered_df["Anomaly"] == -1]
    ax.scatter(
        normal.index,
        normal["Sales"],
        label="Normal",
        alpha=0.6
    )
    ax.scatter(
        anomaly.index,
        anomaly["Sales"],
        color="red",
        label="Anomaly"
    )
    ax.legend()
    st.pyplot(fig)

forecast_df = pd.DataFrame({
    "Forecast Month": future_dates.strftime("%B %Y"),
    "Predicted Sales": future_sales.round(2)
})

st.subheader("Forecast Results")

st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

# Summary

st.subheader("Dashboard Summary")

st.write("""
This dashboard provides an overview of sales performance using the Superstore dataset.

Features included:

- Sales KPIs
- Monthly Sales Trend
- Category-wise Sales
- Region-wise Sales
- Top Selling Products
- Shipping Performance
- Product Clustering
- Sales Anomaly Detection

The dashboard helps business managers monitor performance and identify important sales patterns for better decision making.
""")

