import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(
    layout="wide",
    page_title="The Enterprise Equity Terminal",
    page_icon="📊"
)

# 2. Premium Theme CSS Injection
st.markdown("""
    <style>
    /* Styling for the main header card */
    .terminal-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .terminal-header h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        background: linear-gradient(to right, #38bdf8, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .terminal-header p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Metrics panel card styling */
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* Custom tab and text styles */
    .explanation-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Main Title & Executive Summary
st.markdown("""
    <div class="terminal-header">
        <h1>The Enterprise Equity Terminal</h1>
        <p>Advanced Business Analytics & Information Technology (BAIT) Portfolio Tool. Real-time asset comparison, multi-currency normalization, and predictive analysis.</p>
    </div>
""", unsafe_allow_html=True)

# 4. Sidebar Configuration
st.sidebar.markdown("### ⚙️ Terminal Settings")

# Feature A: Ticker Selection
# Pre-populate list of popular tickers
available_tickers = ["NVDA", "BTC-USD", "META", "MSFT", "AAPL", "AMZN", "GOOGL", "TSLA", "ETH-USD", "SPY", "QQQ"]

custom_ticker = st.sidebar.text_input("➕ Add custom ticker (e.g. AMD, SOL-USD):").upper().strip()
if custom_ticker and custom_ticker not in available_tickers:
    available_tickers.append(custom_ticker)

selected_tickers = st.sidebar.multiselect(
    "Select Assets for Analysis",
    options=available_tickers,
    default=["NVDA", "BTC-USD"],
    help="Select tickers to compare. Use text input above to add custom tickers."
)

# Feature B: Currency Normalization selection
base_currency = st.sidebar.selectbox(
    "Base Currency Normalization",
    options=["USD", "EUR", "GBP", "JPY"],
    index=0,
    help="Mathematically convert all assets to selected base currency using live forex rates."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Chart Configuration")

# Toggles for overlays
show_sma = st.sidebar.checkbox("Show 50-Day SMA", value=True, help="Overlay 50-day Simple Moving Average.")
show_projection = st.sidebar.checkbox("Show 30-Day Projected Path", value=True, help="Overlay 30-day linear regression trendline.")

# Toggle for scale normalization
normalize_chart = st.sidebar.checkbox("Normalize Scale (Base 100)", value=False, help="Normalize assets to start at 100, allowing direct comparison of percentage gains.")

# Lookback window selection for linear regression fit
fit_window = st.sidebar.selectbox(
    "Regression Fit Window",
    options=["Full Year", "Last 90 Days", "Last 30 Days"],
    index=0,
    help="Choose the subset of historical data used to train the linear regression projection."
)

# 5. Data Fetching & Caching
@st.cache_data(ttl=1800)
def fetch_asset_data(ticker_symbol):
    """Fetch 1 year of historical closing prices for a given ticker."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        if df.empty:
            return pd.Series(dtype='float64')
        df.index = df.index.tz_localize(None)
        return df['Close']
    except Exception:
        return pd.Series(dtype='float64')

@st.cache_data(ttl=3600)
def get_forex_rate(target_currency):
    """Fetch live forex exchange rate conversion multiplier from USD to target currency."""
    if target_currency == "USD":
        return 1.0
    
    ticker_map = {
        "EUR": ("EURUSD=X", True),
        "GBP": ("GBPUSD=X", True),
        "JPY": ("USDJPY=X", False)
    }
    
    ticker_name, invert = ticker_map.get(target_currency, (None, False))
    if not ticker_name:
        return 1.0
    
    try:
        ticker = yf.Ticker(ticker_name)
        hist = ticker.history(period="5d")
        if not hist.empty:
            latest_rate = hist['Close'].iloc[-1]
            return 1.0 / latest_rate if invert else latest_rate
        else:
            fallbacks = {"EUR": 0.92, "GBP": 0.79, "JPY": 160.0}
            return fallbacks.get(target_currency, 1.0)
    except Exception:
        fallbacks = {"EUR": 0.92, "GBP": 0.79, "JPY": 160.0}
        return fallbacks.get(target_currency, 1.0)

# Fetch Exchange Rate
forex_rate = get_forex_rate(base_currency)
currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
currency_symbol = currency_symbols.get(base_currency, "$")

# Load Asset Data
if not selected_tickers:
    st.info("ℹ️ Please select at least one asset in the sidebar configuration to begin.")
else:
    data_dict = {}
    failed_tickers = []
    
    with st.spinner("Fetching historical market data..."):
        for ticker in selected_tickers:
            series = fetch_asset_data(ticker)
            if not series.empty:
                data_dict[ticker] = series
            else:
                failed_tickers.append(ticker)
    
    if failed_tickers:
        st.warning(f"⚠️ Failed to retrieve historical data for: {', '.join(failed_tickers)}. Check ticker symbols.")
        
    if not data_dict:
        st.error("❌ No data successfully retrieved for the selected assets.")
    else:
        # Align all dates and interpolate missing values (handling weekend gap between Crypto and Equities)
        df_raw = pd.DataFrame(data_dict)
        df_raw = df_raw.ffill().bfill()
        
        # Apply Currency Normalization
        df_normalized = df_raw * forex_rate
        
        # Create Key Metrics Row
        st.markdown("### 📊 Live Summary Metrics")
        metrics_cols = st.columns(len(df_normalized.columns))
        
        for idx, col_name in enumerate(df_normalized.columns):
            with metrics_cols[idx]:
                latest_val = df_normalized[col_name].iloc[-1]
                prev_val = df_normalized[col_name].iloc[-2] if len(df_normalized) > 1 else latest_val
                delta_val = latest_val - prev_val
                delta_pct = (delta_val / prev_val) * 100 if prev_val != 0 else 0.0
                
                with st.container(border=True):
                    st.metric(
                        label=f"{col_name} Price ({base_currency})",
                        value=f"{currency_symbol}{latest_val:,.2f}" if base_currency != "JPY" else f"{currency_symbol}{latest_val:,.0f}",
                        delta=f"{delta_val:+.2f} ({delta_pct:+.2f}%)" if base_currency != "JPY" else f"{delta_val:+.0f} ({delta_pct:+.2f}%)"
                    )

        # 6. Interactive Price & Projection Chart Section
        st.markdown("### 📈 Interactive Price & Technical Analysis Chart")
        st.markdown(
            f"<div class='explanation-box'>"
            f"<strong>Technical Insight:</strong> This chart displays the 1-year historical performance in "
            f"<strong>{base_currency}</strong>. The 50-day Simple Moving Average (SMA) tracks medium-term momentum. "
            f"The 30-day Projected Path is generated via an OLS linear regression model trained on the "
            f"selected <strong>{fit_window}</strong> fit window."
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Build Chart Data
        df_chart = df_normalized.copy()
        if normalize_chart:
            for col in df_chart.columns:
                first_val = df_chart[col].iloc[0]
                if first_val != 0:
                    df_chart[col] = (df_chart[col] / first_val) * 100
        
        # Plotly Figure
        fig = go.Figure()
        
        # Colors mapping
        COLOR_PALETTE = ["#10b981", "#f59e0b", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444", "#f97316"]
        colors = {ticker: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, ticker in enumerate(df_chart.columns)}
        
        # Draw lines for each asset
        for ticker in df_chart.columns:
            # Historical Prices
            fig.add_trace(go.Scatter(
                x=df_chart.index,
                y=df_chart[ticker],
                mode='lines',
                name=f"{ticker} (Price)",
                line=dict(color=colors[ticker], width=2),
                hovertemplate='%{x}<br>Price: %{y:,.2f}'
            ))
            
            # SMA Overlay
            if show_sma:
                sma_series = df_chart[ticker].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=sma_series.index,
                    y=sma_series,
                    mode='lines',
                    name=f"{ticker} (SMA 50)",
                    line=dict(color=colors[ticker], width=1.5, dash='dash'),
                    hovertemplate='%{x}<br>SMA 50: %{y:,.2f}'
                ))
            
            # Linear Regression Projection Overlay
            if show_projection:
                y_series = df_chart[ticker]
                N = len(y_series)
                
                # Filter series for regression training based on user selection
                if fit_window == "Last 90 Days":
                    fit_series = y_series.tail(90)
                elif fit_window == "Last 30 Days":
                    fit_series = y_series.tail(30)
                else:
                    fit_series = y_series
                
                M = len(fit_series)
                fit_indices = np.arange(N - M, N)
                
                # Fit OLS model
                slope, intercept = np.polyfit(fit_indices, fit_series.values, 1)
                
                # Project 30 days out from the last data point
                proj_indices = np.arange(N - 1, N + 30)
                proj_values = slope * proj_indices + intercept
                
                # Projection Dates
                last_date = y_series.index[-1]
                proj_dates = [last_date + timedelta(days=i) for i in range(31)]
                
                fig.add_trace(go.Scatter(
                    x=proj_dates,
                    y=proj_values,
                    mode='lines',
                    name=f"{ticker} (Projected)",
                    line=dict(color=colors[ticker], width=1.5, dash='dot'),
                    hovertemplate='%{x}<br>Projected: %{y:,.2f}'
                ))
        
        # Configure Layout
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Indexed Price (Base 100)" if normalize_chart else f"Price ({base_currency})",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 7. Options Chain Intelligence Section
        st.markdown("---")
        st.markdown("### ⛓️ Options Chain Intelligence")
        
        # Identify tickers that support options (non-crypto/forex assets)
        @st.cache_data(ttl=1800)
        def filter_options_tickers(tickers):
            options_avail = []
            for t in tickers:
                try:
                    obj = yf.Ticker(t)
                    if obj.options:
                        options_avail.append(t)
                except Exception:
                    pass
            return options_avail
            
        options_tickers = filter_options_tickers(df_normalized.columns.tolist())
        
        if not options_tickers:
            st.info("ℹ️ Options chains are not available for the selected assets (cryptocurrencies, forex, and indexes generally do not support standard equities option chain downloads).")
        else:
            opt_col1, opt_col2 = st.columns([1, 2])
            
            with opt_col1:
                selected_opt_ticker = st.selectbox(
                    "Select Asset for Options Data",
                    options=options_tickers,
                    help="Choose an asset to view its derivatives option chain."
                )
                
                ticker_obj = yf.Ticker(selected_opt_ticker)
                expirations = ticker_obj.options
                
                if expirations:
                    selected_exp = st.selectbox(
                        "Select Expiration Date",
                        options=expirations,
                        help="Choose the option contract expiration date."
                    )
                else:
                    selected_exp = None
                    st.warning("No expiration dates found for this asset.")
            
            with opt_col2:
                st.markdown(
                    f"<div class='explanation-box' style='margin-top: 25px;'>"
                    f"<strong>Derivatives Analysis:</strong> Below is the options table showing Calls and Puts "
                    f"for <strong>{selected_opt_ticker}</strong> expiring on <strong>{selected_exp}</strong>. "
                    f"Strike prices and contract last prices are normalized to <strong>{base_currency}</strong>. "
                    f"Implied Volatility (IV) represents forward-looking market expectations of asset volatility."
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            if selected_exp:
                with st.spinner("Fetching option chain..."):
                    try:
                        chain = ticker_obj.option_chain(selected_exp)
                        
                        # Extract columns
                        calls = chain.calls[['strike', 'lastPrice', 'impliedVolatility']].copy()
                        puts = chain.puts[['strike', 'lastPrice', 'impliedVolatility']].copy()
                        
                        # Format Columns
                        calls.columns = ['Strike Price', 'Last Price', 'Implied Volatility (IV)']
                        puts.columns = ['Strike Price', 'Last Price', 'Implied Volatility (IV)']
                        
                        # Apply exchange rate normalization to Option strikes & prices
                        if base_currency != "USD":
                            calls['Strike Price'] *= forex_rate
                            calls['Last Price'] *= forex_rate
                            puts['Strike Price'] *= forex_rate
                            puts['Last Price'] *= forex_rate
                        
                        # Convert Implied Volatility to percentage
                        calls['Implied Volatility (IV)'] *= 100
                        puts['Implied Volatility (IV)'] *= 100
                        
                        # Render Tables Side-by-Side
                        c_col, p_col = st.columns(2)
                        
                        with c_col:
                            st.markdown("##### 🟢 Call Options")
                            st.dataframe(
                                calls,
                                column_config={
                                    "Strike Price": st.column_config.NumberColumn(format=f"{currency_symbol}%.2f"),
                                    "Last Price": st.column_config.NumberColumn(format=f"{currency_symbol}%.2f"),
                                    "Implied Volatility (IV)": st.column_config.NumberColumn(format="%.2f%%"),
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            
                        with p_col:
                            st.markdown("##### 🔴 Put Options")
                            st.dataframe(
                                puts,
                                column_config={
                                    "Strike Price": st.column_config.NumberColumn(format=f"{currency_symbol}%.2f"),
                                    "Last Price": st.column_config.NumberColumn(format=f"{currency_symbol}%.2f"),
                                    "Implied Volatility (IV)": st.column_config.NumberColumn(format="%.2f%%"),
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                    except Exception as e:
                        st.error(f"❌ Error downloading option chain: {e}")
