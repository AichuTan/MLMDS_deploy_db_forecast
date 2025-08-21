import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from supabase import create_client, Client
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized

# =========================
# DB CONNECTION (resource)
# =========================
@st.cache_resource
def init_connection() -> Client:
    url: str = st.secrets["supabase_url"]
    key: str = st.secrets["supabase_key"]
    return create_client(url, key)

supabase = init_connection()

# =========================
# DATA FETCH (cache data)
# =========================
@st.cache_data(ttl=600)  # cache clears after 10 minutes
def run_query() -> list[dict]:
    # Return only serializable data (list[dict]), not the APIResponse object
    resp = supabase.table("car_parts_monthly_sale").select("*").execute()
    return resp.data

@st.cache_data(ttl=600)
def create_dataframe() -> pd.DataFrame:
    rows = run_query()
    df = pd.json_normalize(rows)

    # Ensure expected columns exist; adjust names if your schema differs
    # Expected: id, parts_id, date, volume
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    # Sort for plotting/forecasting
    if {"parts_id", "date"}.issubset(df.columns):
        df = df.sort_values(["parts_id", "date"]).reset_index(drop=True)

    return df

# =========================
# PLOTTING (no cache)
# =========================
def plot_volume(df: pd.DataFrame, ids: list[int] | list[str]) -> None:
    if not ids:
        st.info("Select at least one product ID to see its history.")
        return

    missing_cols = {"parts_id", "date", "volume"} - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns for plot: {missing_cols}")
        return

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(10))

    for pid in ids:
        sub = df[df["parts_id"] == pid]
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["volume"], label=str(pid))

    ax.legend(loc="best")
    fig.autofmt_xdate()
    ax.set_xlabel("Month")
    ax.set_ylabel("Volume")
    ax.set_title("Monthly Volume by Product")
    st.pyplot(fig)

# =========================
# PREPARE DATA FOR MODEL
# =========================
def format_dataset(df: pd.DataFrame, ids: list[int] | list[str]) -> pd.DataFrame:
    needed = {"parts_id", "date", "volume"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns for model: {needed - set(df.columns)}")

    model_df = df[df["parts_id"].isin(ids)].copy()
    # StatsForecast expects: unique_id, ds (datetime), y (numeric)
    model_df = (
        model_df.rename(columns={"parts_id": "unique_id", "date": "ds", "volume": "y"})
                 .loc[:, ["unique_id", "ds", "y"]]
    )
    # Ensure correct dtypes
    model_df["ds"] = pd.to_datetime(model_df["ds"])
    model_df["y"] = pd.to_numeric(model_df["y"], errors="coerce").fillna(0.0)
    return model_df

# =========================
# FORECASTING (build & run)
# =========================
def make_predictions(df: pd.DataFrame, ids: list[int] | list[str], horizon: int) -> pd.DataFrame:
    model_df = format_dataset(df, ids)
    models = [CrostonOptimized()]  # good for intermittent demand

    sf = StatsForecast(
    models=models,
    freq="MS",
    n_jobs=-1
    )
    forecast_df = sf.forecast(df=model_df, h=horizon)

    return forecast_df

# =========================
# APP
# =========================
if __name__ == "__main__":
    st.title("Forecast product demand")

    df = create_dataframe()

    st.subheader("Select a product")
    product_ids = st.multiselect(
        "Select product ID",
        options=sorted(df["parts_id"].dropna().unique().tolist())
    )

    # Plot history
    plot_volume(df, product_ids)

    with st.expander("Forecast"):
        if len(product_ids) == 0:
            st.warning("Select at least one product ID to forecast")
        else:
            horizon = st.slider("Horizon (months ahead)", 1, 12, value=6, step=1)

            if st.button("Forecast", type="primary"):
                with st.spinner("Making predictions..."):
                    fcst = make_predictions(df, product_ids, horizon)

                # Show a preview
                st.subheader("Forecast preview")
                st.dataframe(fcst, use_container_width=True)

                # Offer download
                csv_file = fcst.to_csv(index=False)
                st.download_button(
                    label="Download predictions",
                    data=csv_file,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
