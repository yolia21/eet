import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize FastAPI App
app = FastAPI(title="The Enterprise Equity Terminal")

def fetch_options_data(ticker_symbol: str, target_date: Optional[str], forex_rate: float):
    """Fetch calls and puts for a given ticker and date, and normalize pricing to base currency."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = list(ticker.options)
        if not expirations:
            return {
                "ticker": ticker_symbol,
                "expiration_dates": [],
                "selected_date": None,
                "calls": [],
                "puts": [],
                "error": "No options available for this ticker."
            }
        
        # Default to first expiration if not specified
        selected_date = target_date if target_date in expirations else expirations[0]
        chain = ticker.option_chain(selected_date)
        
        # Process and convert Calls
        calls_df = chain.calls[['strike', 'lastPrice', 'impliedVolatility']].copy()
        calls_df['strike'] *= forex_rate
        calls_df['lastPrice'] *= forex_rate
        calls_df['impliedVolatility'] *= 100
        calls_list = calls_df.replace({np.nan: None}).to_dict(orient="records")
        
        # Process and convert Puts
        puts_df = chain.puts[['strike', 'lastPrice', 'impliedVolatility']].copy()
        puts_df['strike'] *= forex_rate
        puts_df['lastPrice'] *= forex_rate
        puts_df['impliedVolatility'] *= 100
        puts_list = puts_df.replace({np.nan: None}).to_dict(orient="records")
        
        return {
            "ticker": ticker_symbol,
            "expiration_dates": expirations,
            "selected_date": selected_date,
            "calls": calls_list,
            "puts": puts_list,
            "error": None
        }
    except Exception as e:
        return {
            "ticker": ticker_symbol,
            "expiration_dates": [],
            "selected_date": None,
            "calls": [],
            "puts": [],
            "error": f"Failed to retrieve options: {str(e)}"
        }

def fetch_market_data(tickers_list: List[str], currency: str, fit_window: str):
    """Fetch 1 year of historical closing prices, apply forex normalizations, and compute technical paths."""
    forex_rate = 1.0
    
    # Fetch Forex Rate
    if currency != "USD":
        ticker_map = {
            "EUR": ("EURUSD=X", True),
            "GBP": ("GBPUSD=X", True),
            "JPY": ("USDJPY=X", False)
        }
        ticker_name, invert = ticker_map.get(currency, (None, False))
        if ticker_name:
            try:
                forex_ticker = yf.Ticker(ticker_name)
                hist = forex_ticker.history(period="5d")
                if not hist.empty:
                    rate_val = hist['Close'].iloc[-1]
                    forex_rate = 1.0 / rate_val if invert else rate_val
            except Exception:
                fallbacks = {"EUR": 0.92, "GBP": 0.79, "JPY": 160.0}
                forex_rate = fallbacks.get(currency, 1.0)
                
    currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    currency_symbol = currency_symbols.get(currency, "$")
    
    data_payload = {}
    valid_tickers = []
    data_dict = {}
    
    # Fetch assets individually
    for t in tickers_list:
        try:
            ticker_obj = yf.Ticker(t)
            hist = ticker_obj.history(period="1y")
            if not hist.empty:
                hist.index = hist.index.tz_localize(None)
                data_dict[t] = hist['Close']
        except Exception:
            pass
            
    if data_dict:
        # Combine into DataFrame to align dates and handle gaps
        df = pd.DataFrame(data_dict)
        df = df.ffill().bfill()
        
        # Apply currency normalization
        df_norm = df * forex_rate
        dates_list = df_norm.index.strftime('%Y-%m-%d').tolist()
        
        for t in df_norm.columns:
            series = df_norm[t]
            prices = series.values
            N = len(series)
            
            # Metric details
            latest_price = float(prices[-1])
            prev_price = float(prices[-2]) if N > 1 else latest_price
            change_val = latest_price - prev_price
            change_pct = (change_val / prev_price) * 100 if prev_price != 0 else 0.0
            
            # 50-day Simple Moving Average (SMA)
            sma50 = series.rolling(window=50).mean().values
            
            # Regression Projection (OLS)
            if fit_window == "Last 90 Days":
                fit_series = series.tail(90)
            elif fit_window == "Last 30 Days":
                fit_series = series.tail(30)
            else:
                fit_series = series
                
            M = len(fit_series)
            fit_indices = np.arange(N - M, N)
            
            try:
                slope, intercept = np.polyfit(fit_indices, fit_series.values, 1)
                proj_indices = np.arange(N - 1, N + 30)
                proj_prices = slope * proj_indices + intercept
                
                last_date = series.index[-1]
                proj_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
            except Exception:
                proj_prices = np.array([])
                proj_dates = []
                
            # Serialize
            data_payload[t] = {
                "dates": dates_list,
                "prices": [None if pd.isna(p) else float(p) for p in prices],
                "sma50": [None if pd.isna(s) else float(s) for s in sma50],
                "projection_dates": proj_dates,
                "projection_prices": [None if pd.isna(p) else float(p) for p in proj_prices],
                "latest_price": latest_price,
                "prev_price": prev_price,
                "change_val": change_val,
                "change_pct": change_pct
            }
            valid_tickers.append(t)
            
    return {
        "forex_rate": forex_rate,
        "currency_symbol": currency_symbol,
        "tickers": valid_tickers,
        "data": data_payload
    }

# ----------------- API ROUTES -----------------

@app.get("/api/market-data")
def get_market_data(
    tickers: str = Query("NVDA,BTC-USD", description="Comma-separated ticker symbols"),
    currency: str = Query("USD", description="Base display currency (USD, EUR, GBP, JPY)"),
    fit_window: str = Query("Full Year", description="OLS regression lookback fit window"),
    options_ticker: Optional[str] = Query(None, description="Active ticker for options analysis"),
    options_date: Optional[str] = Query(None, description="Selected option contract expiration date")
):
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="At least one ticker symbol is required.")
        
    market_result = fetch_market_data(tickers_list, currency, fit_window)
    
    options_result = None
    if options_ticker:
        opt_t = options_ticker.strip().upper()
        options_result = fetch_options_data(opt_t, options_date, market_result["forex_rate"])
        
    return {
        "forex_rate": market_result["forex_rate"],
        "currency_symbol": market_result["currency_symbol"],
        "tickers": market_result["tickers"],
        "data": market_result["data"],
        "options": options_result
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Enterprise Equity Terminal</title>
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            --bg-base: #070a13;
            --bg-card: rgba(13, 20, 38, 0.65);
            --border-glow: rgba(56, 189, 248, 0.15);
            --border-faint: rgba(255, 255, 255, 0.05);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-glow: #38bdf8;
            --accent-green: #10b981;
            --accent-red: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            background-image: radial-gradient(circle at 10% 20%, rgba(30, 41, 59, 0.15) 0%, transparent 80%),
                              radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.05) 0%, transparent 70%);
            background-attachment: fixed;
            min-height: 100vh;
            padding: 24px;
            padding-bottom: 60px;
        }

        .terminal-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 12px 24px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: rgba(7, 10, 19, 0.9);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-top: 1px solid var(--border-faint);
            z-index: 1000;
        }

        /* Layout Grid */
        .terminal-container {
            max-width: 1600px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 24px;
        }

        /* Glassmorphism Panel styles */
        .panel {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-faint);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .panel:hover {
            border-color: var(--border-glow);
            box-shadow: 0 8px 32px rgba(56, 189, 248, 0.05);
        }

        /* Header Card */
        .header-panel {
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 32px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        }

        .header-title-section h1 {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2rem;
            background: linear-gradient(to right, #38bdf8, #3b82f6, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .header-title-section p {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .terminal-status {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 6px 12px;
            border-radius: 20px;
            color: var(--accent-green);
            font-size: 0.8rem;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
        }

        /* Sidebar Controls */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            border-bottom: 1px solid var(--border-faint);
            padding-bottom: 8px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
        }

        .control-group label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        /* Custom Dropdowns */
        select, input[type="text"] {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-faint);
            padding: 10px 14px;
            border-radius: 8px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.3s ease;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.1);
        }

        /* Custom Checkbox styles */
        .checkbox-container {
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            font-size: 0.85rem;
            color: var(--text-secondary);
            padding: 4px 0;
            user-select: none;
        }

        .checkbox-container input {
            display: none;
        }

        .checkbox-checkmark {
            width: 18px;
            height: 18px;
            border: 1.5px solid var(--border-faint);
            border-radius: 4px;
            display: inline-block;
            position: relative;
            background: rgba(15, 23, 42, 0.6);
            transition: border-color 0.2s, background-color 0.2s;
        }

        .checkbox-container input:checked + .checkbox-checkmark {
            background-color: var(--accent-glow);
            border-color: var(--accent-glow);
        }

        .checkbox-container input:checked + .checkbox-checkmark::after {
            content: "";
            position: absolute;
            left: 5px;
            top: 2px;
            width: 5px;
            height: 9px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }

        /* Add Ticker Section */
        .ticker-item-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-faint);
            padding: 8px;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.3);
        }

        .add-ticker-row {
            display: flex;
            gap: 8px;
        }

        .btn-add {
            background: var(--accent-glow);
            border: none;
            padding: 0 16px;
            border-radius: 8px;
            color: #070a13;
            font-weight: 700;
            cursor: pointer;
            transition: opacity 0.2s;
        }

        .btn-add:hover {
            opacity: 0.9;
        }

        /* Main Content */
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Summary Metrics */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-faint);
            border-radius: 10px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }

        .metric-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--accent-glow);
        }

        .metric-card.positive::after {
            background-color: var(--accent-green);
        }

        .metric-card.negative::after {
            background-color: var(--accent-red);
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 8px;
        }

        .metric-delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .metric-delta.positive {
            color: var(--accent-green);
        }

        .metric-delta.negative {
            color: var(--accent-red);
        }

        /* Chart Canvas Card */
        .chart-panel {
            position: relative;
            min-height: 450px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-title h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
        }

        .chart-insight {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Options Grid Section */
        .options-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .options-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-faint);
            padding-bottom: 12px;
        }

        .options-config {
            display: flex;
            gap: 16px;
            align-items: center;
        }

        .options-tables-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .options-table-card h3 {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .table-wrapper {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border-faint);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.3);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th, td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-faint);
        }

        th {
            background: rgba(15, 23, 42, 0.8);
            font-weight: 600;
            color: var(--text-secondary);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        /* Loading Spinner Overlays */
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(7, 10, 19, 0.8);
            z-index: 100;
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 12px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            width: 48px;
            height: 48px;
            border: 4px solid var(--border-faint);
            border-top-color: var(--accent-glow);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Info box styling */
        .info-msg {
            padding: 24px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Scrollbars styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Mobile adaptation */
        @media (max-width: 900px) {
            .terminal-container {
                grid-template-columns: 1fr;
            }
            .options-tables-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>

    <div class="terminal-container">
        
        <!-- Header -->
        <header class="panel header-panel">
            <div class="header-title-section">
                <h1>The Enterprise Equity Terminal</h1>
                <p>Advanced BAIT Financial Analytics Dashboard • Served via FastAPI Backend</p>
            </div>
            <div class="terminal-status">
                <div class="status-dot"></div>
                <span>FASTAPI SECURE LOCALHOST</span>
            </div>
        </header>

        <!-- Sidebar Config -->
        <aside class="sidebar">
            
            <div class="panel">
                <div class="section-title">📊 Asset Selector</div>
                
                <div class="control-group">
                    <label>Manage Active Assets</label>
                    <div class="ticker-item-list" id="tickerCheckboxList">
                        <!-- Populated by JavaScript -->
                    </div>
                </div>

                <div class="control-group">
                    <div class="add-ticker-row">
                        <input type="text" id="addTickerInput" placeholder="e.g. AMD, SOL-USD" style="text-transform: uppercase;">
                        <button class="btn-add" id="btnAddTicker">+</button>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="section-title">⚙️ Terminal Config</div>

                <div class="control-group">
                    <label>Base Currency Normalization</label>
                    <select id="currencySelect">
                        <option value="USD">USD ($)</option>
                        <option value="EUR">EUR (€)</option>
                        <option value="GBP">GBP (£)</option>
                        <option value="JPY">JPY (¥)</option>
                    </select>
                </div>

                <div class="control-group" style="margin-top: 10px;">
                    <label class="checkbox-container">
                        <input type="checkbox" id="normalizeCheck">
                        <span class="checkbox-checkmark"></span>
                        Normalize Scale (Base 100)
                    </label>
                </div>

                <div class="control-group" style="margin-top: 10px;">
                    <label class="checkbox-container">
                        <input type="checkbox" id="smaCheck" checked>
                        <span class="checkbox-checkmark"></span>
                        Show 50-Day SMA
                    </label>
                </div>

                <div class="control-group">
                    <label class="checkbox-container">
                        <input type="checkbox" id="projectionCheck" checked>
                        <span class="checkbox-checkmark"></span>
                        Show 30-Day Projection
                    </label>
                </div>

                <div class="control-group">
                    <label>Regression Fit Window</label>
                    <select id="fitWindowSelect">
                        <option value="Full Year">Full Year</option>
                        <option value="Last 90 Days">Last 90 Days</option>
                        <option value="Last 30 Days">Last 30 Days</option>
                    </select>
                </div>
            </div>

        </aside>

        <!-- Main Workspace -->
        <main class="main-content">
            
            <!-- Live Summary Metrics Row -->
            <div class="metrics-grid" id="metricsGrid">
                <!-- Populated dynamically -->
            </div>

            <!-- Price Analysis Chart Panel -->
            <div class="panel chart-panel">
                <div class="loading-overlay" id="chartLoader">
                    <div class="spinner"></div>
                </div>
                
                <div class="chart-header">
                    <div class="chart-title">
                        <h2>Closing Prices & Projected Path</h2>
                        <p class="chart-insight" id="chartSubtitle">1-Year Historical Daily Series • Click legend to hide/show specific assets</p>
                    </div>
                </div>
                
                <div style="height: 380px; width: 100%; position: relative;">
                    <canvas id="marketChartCanvas"></canvas>
                </div>
            </div>

            <!-- Options Chain Panel -->
            <div class="panel options-panel">
                <div class="loading-overlay" id="optionsLoader">
                    <div class="spinner"></div>
                </div>

                <div class="options-header-row">
                    <div class="chart-title">
                        <h2>⛓️ Options Chain Intelligence</h2>
                        <p class="chart-insight">Derivatives grids showing Calls and Puts side-by-side with strikes/prices normalized.</p>
                    </div>
                    
                    <div class="options-config">
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            <label style="font-size: 0.75rem; color: var(--text-secondary);">Asset</label>
                            <select id="optionsTickerSelect" style="width: 140px; padding: 6px 12px;"></select>
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 4px;">
                            <label style="font-size: 0.75rem; color: var(--text-secondary);">Expiration Date</label>
                            <select id="optionsDateSelect" style="width: 160px; padding: 6px 12px;"></select>
                        </div>
                    </div>
                </div>

                <div id="optionsOutputArea">
                    <div class="options-tables-container">
                        <!-- Calls -->
                        <div class="options-table-card">
                            <h3 style="color: var(--accent-green);">🟢 Call Options</h3>
                            <div class="table-wrapper">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Strike Price</th>
                                            <th>Last Price</th>
                                            <th>Implied Volatility (IV)</th>
                                        </tr>
                                    </thead>
                                    <tbody id="callsTableBody"></tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Puts -->
                        <div class="options-table-card">
                            <h3 style="color: var(--accent-red);">🔴 Put Options</h3>
                            <div class="table-wrapper">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Strike Price</th>
                                            <th>Last Price</th>
                                            <th>Implied Volatility (IV)</th>
                                        </tr>
                                    </thead>
                                    <tbody id="putsTableBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

            </div>

        </main>
    </div>

    <!-- Frontend JS Engine -->
    <script>
        // Terminal Application State
        const state = {
            tickers: ["NVDA", "BTC-USD", "META", "ETH-USD"],
            activeTickers: ["NVDA", "BTC-USD"],
            currency: "USD",
            normalizeScale: false,
            showSMA: true,
            showProjection: true,
            fitWindow: "Full Year",
            optionsTicker: "NVDA",
            optionsDate: "",
            
            // Server data caches
            marketData: null,
            forexRate: 1.0,
            currencySymbol: "$"
        };

        // UI Element References
        const tickerCheckboxList = document.getElementById("tickerCheckboxList");
        const addTickerInput = document.getElementById("addTickerInput");
        const btnAddTicker = document.getElementById("btnAddTicker");
        const currencySelect = document.getElementById("currencySelect");
        const normalizeCheck = document.getElementById("normalizeCheck");
        const smaCheck = document.getElementById("smaCheck");
        const projectionCheck = document.getElementById("projectionCheck");
        const fitWindowSelect = document.getElementById("fitWindowSelect");
        const metricsGrid = document.getElementById("metricsGrid");
        const optionsTickerSelect = document.getElementById("optionsTickerSelect");
        const optionsDateSelect = document.getElementById("optionsDateSelect");
        const callsTableBody = document.getElementById("callsTableBody");
        const putsTableBody = document.getElementById("putsTableBody");
        const chartLoader = document.getElementById("chartLoader");
        const optionsLoader = document.getElementById("optionsLoader");
        
        let chartInstance = null;

        // Initialize UI Elements
        function initUI() {
            renderTickerCheckboxes();
            
            // Wire listeners
            currencySelect.addEventListener("change", (e) => {
                state.currency = e.target.value;
                triggerTerminalUpdate();
            });
            
            normalizeCheck.addEventListener("change", (e) => {
                state.normalizeScale = e.target.checked;
                renderChart(); // local toggle (fast client update)
            });

            smaCheck.addEventListener("change", (e) => {
                state.showSMA = e.target.checked;
                renderChart(); // local toggle
            });

            projectionCheck.addEventListener("change", (e) => {
                state.showProjection = e.target.checked;
                renderChart(); // local toggle
            });

            fitWindowSelect.addEventListener("change", (e) => {
                state.fitWindow = e.target.value;
                triggerTerminalUpdate();
            });

            btnAddTicker.addEventListener("click", addCustomTicker);
            addTickerInput.addEventListener("keypress", (e) => {
                if (e.key === "Enter") addCustomTicker();
            });

            optionsTickerSelect.addEventListener("change", (e) => {
                state.optionsTicker = e.target.value;
                state.optionsDate = ""; // reset to first date
                triggerOptionsUpdate();
            });

            optionsDateSelect.addEventListener("change", (e) => {
                state.optionsDate = e.target.value;
                triggerOptionsUpdate();
            });
        }

        // Render Sidebar Ticker Checkboxes
        function renderTickerCheckboxes() {
            tickerCheckboxList.innerHTML = "";
            state.tickers.forEach(ticker => {
                const label = document.createElement("label");
                label.className = "checkbox-container";
                
                const checked = state.activeTickers.includes(ticker) ? "checked" : "";
                label.innerHTML = `
                    <input type="checkbox" value="${ticker}" ${checked} onchange="toggleTicker(this)">
                    <span class="checkbox-checkmark"></span>
                    ${ticker}
                `;
                tickerCheckboxList.appendChild(label);
            });
        }

        // Add Ticker from Input
        function addCustomTicker() {
            const val = addTickerInput.value.trim().toUpperCase();
            if (val && !state.tickers.includes(val)) {
                state.tickers.push(val);
                state.activeTickers.push(val);
                renderTickerCheckboxes();
                addTickerInput.value = "";
                triggerTerminalUpdate();
            }
        }

        // Handle Checkbox Toggling
        window.toggleTicker = function(checkbox) {
            const val = checkbox.value;
            if (checkbox.checked) {
                if (!state.activeTickers.includes(val)) state.activeTickers.push(val);
            } else {
                state.activeTickers = state.activeTickers.filter(t => t !== val);
            }
            
            // Maintain active equity for option chains dynamically
            updateOptionsTickerDropdown();
            triggerTerminalUpdate();
        };

        // Populate options ticker selector with active non-crypto tickers
        function updateOptionsTickerDropdown() {
            const oldVal = state.optionsTicker;
            optionsTickerSelect.innerHTML = "";
            
            // Filter non-crypto assets (e.g. BTC-USD, ETH-USD contains '-')
            const optionable = state.activeTickers.filter(t => !t.includes('-') && !t.includes('='));
            
            if (optionable.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "N/A";
                optionsTickerSelect.appendChild(opt);
                state.optionsTicker = "";
            } else {
                optionable.forEach(t => {
                    const opt = document.createElement("option");
                    opt.value = t;
                    opt.textContent = t;
                    optionsTickerSelect.appendChild(opt);
                });
                if (optionable.includes(oldVal)) {
                    optionsTickerSelect.value = oldVal;
                    state.optionsTicker = oldVal;
                } else {
                    optionsTickerSelect.value = optionable[0];
                    state.optionsTicker = optionable[0];
                }
            }
        }

        // Trigger full data reload
        async function triggerTerminalUpdate() {
            if (state.activeTickers.length === 0) {
                metricsGrid.innerHTML = '<div class="metric-card" style="grid-column: 1/-1;"><div class="info-msg">Please select at least one active asset.</div></div>';
                if (chartInstance) chartInstance.destroy();
                clearOptionsGrids();
                return;
            }

            chartLoader.classList.add("active");
            
            try {
                const tickersQuery = state.activeTickers.join(",");
                let url = `/api/market-data?tickers=${tickersQuery}&currency=${state.currency}&fit_window=${state.fitWindow}`;
                
                // Add option chain request details
                if (state.optionsTicker) {
                    url += `&options_ticker=${state.optionsTicker}&options_date=${state.optionsDate}`;
                }

                const response = await fetch(url);
                const data = await response.json();
                
                if (response.ok) {
                    state.marketData = data.data;
                    state.forexRate = data.forex_rate;
                    state.currencySymbol = data.currency_symbol;
                    
                    renderMetrics(data.data);
                    renderChart();
                    
                    // Manage options returns
                    if (data.options) {
                        renderOptionsDropdown(data.options);
                        renderOptionsTables(data.options);
                    } else {
                        clearOptionsGrids();
                    }
                } else {
                    console.error("API error:", data.detail);
                }
            } catch (err) {
                console.error("Fetch failed:", err);
            } finally {
                chartLoader.classList.remove("active");
            }
        }

        // Trigger only option chain update (faster, avoid reloading chart)
        async function triggerOptionsUpdate() {
            if (!state.optionsTicker) {
                clearOptionsGrids();
                return;
            }

            optionsLoader.classList.add("active");
            try {
                const url = `/api/market-data?tickers=${state.optionsTicker}&currency=${state.currency}&fit_window=${state.fitWindow}&options_ticker=${state.optionsTicker}&options_date=${state.optionsDate}`;
                const response = await fetch(url);
                const data = await response.json();
                
                if (response.ok && data.options) {
                    renderOptionsTables(data.options);
                }
            } catch (err) {
                console.error("Options fetch failed:", err);
            } finally {
                optionsLoader.classList.remove("active");
            }
        }

        // Render Top summary metrics
        function renderMetrics(data) {
            metricsGrid.innerHTML = "";
            
            Object.keys(data).forEach(ticker => {
                const tickerData = data[ticker];
                const card = document.createElement("div");
                
                const isPos = tickerData.change_val >= 0;
                card.className = `metric-card ${isPos ? 'positive' : 'negative'}`;
                
                const formattedPrice = formatCurrency(tickerData.latest_price);
                const formattedChange = (isPos ? "+" : "") + formatCurrency(tickerData.change_val);
                const formattedPct = (isPos ? "+" : "") + tickerData.change_pct.toFixed(2) + "%";
                
                card.innerHTML = `
                    <div class="metric-label">${ticker}</div>
                    <div class="metric-value">${formattedPrice}</div>
                    <div class="metric-delta ${isPos ? 'positive' : 'negative'}">
                        <span>${isPos ? '▲' : '▼'}</span>
                        <span>${formattedChange} (${formattedPct})</span>
                    </div>
                `;
                metricsGrid.appendChild(card);
            });
        }

        // Render Chart.js Plot
        function renderChart() {
            const canvas = document.getElementById("marketChartCanvas");
            if (!state.marketData) return;
            
            if (chartInstance) {
                chartInstance.destroy();
            }

            const COLOR_PALETTE = ["#10b981", "#f59e0b", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444", "#f97316"];
            const datasets = [];
            let globalLabels = [];
            
            // Get unique color for each active ticker
            const tickers = Object.keys(state.marketData);
            
            tickers.forEach((ticker, idx) => {
                const tColor = COLOR_PALETTE[idx % COLOR_PALETTE.length];
                const item = state.marketData[ticker];
                
                let baseVal = item.prices[0] || 1.0;
                
                // 1. Raw or Normalized Prices
                const pricesData = state.normalizeScale 
                    ? item.prices.map(p => p !== null ? (p / baseVal) * 100 : null)
                    : item.prices;
                
                globalLabels = item.dates;
                
                // Add Gradient Area Fill under price line
                const ctx = canvas.getContext('2d');
                const gradient = ctx.createLinearGradient(0, 0, 0, 350);
                gradient.addColorStop(0, hexToRgba(tColor, 0.15));
                gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

                datasets.push({
                    label: `${ticker} Price`,
                    data: pricesData,
                    borderColor: tColor,
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.15,
                    pointRadius: 0,
                    pointHoverRadius: 5
                });

                // 2. SMA 50 Overlay
                if (state.showSMA && item.sma50) {
                    const smaData = state.normalizeScale
                        ? item.sma50.map(s => s !== null ? (s / baseVal) * 100 : null)
                        : item.sma50;
                        
                    datasets.push({
                        label: `${ticker} SMA 50`,
                        data: smaData,
                        borderColor: tColor,
                        borderWidth: 1.5,
                        borderDash: [6, 4],
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        hidden: false
                    });
                }

                // 3. OLS 30-Day Projection Overlay
                if (state.showProjection && item.projection_prices && item.projection_prices.length > 0) {
                    const projPrices = state.normalizeScale
                        ? item.projection_prices.map(p => p !== null ? (p / baseVal) * 100 : null)
                        : item.projection_prices;
                        
                    // Projections extend beyond index. Add padding dates to labels
                    const extendedLabels = [...globalLabels];
                    
                    // Add projection points aligned to projection dates
                    // Insert nulls for historical length so lines align in chart.js multi-axis
                    const alignedProjData = Array(pricesData.length - 1).fill(null).concat(projPrices);
                    
                    // Make sure projection dates are in labels if not already
                    item.projection_dates.forEach((date, pIdx) => {
                        if (pIdx > 0 && !extendedLabels.includes(date)) {
                            extendedLabels.push(date);
                        }
                    });
                    
                    if (extendedLabels.length > globalLabels.length) {
                        globalLabels = extendedLabels;
                    }

                    datasets.push({
                        label: `${ticker} Projected`,
                        data: alignedProjData,
                        borderColor: tColor,
                        borderWidth: 1.5,
                        borderDash: [2, 3],
                        fill: false,
                        pointRadius: 0,
                        tension: 0.05
                    });
                }
            });

            const subtitleText = state.normalizeScale 
                ? "Normalized relative curves starting at 100"
                : `Prices normalized to Base Currency (${state.currency})`;
            document.getElementById("chartSubtitle").textContent = subtitleText;

            chartInstance = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: globalLabels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#f8fafc',
                                font: {
                                    family: 'Inter',
                                    size: 11
                                },
                                padding: 15
                            }
                        },
                        tooltip: {
                            backgroundColor: '#0f172a',
                            titleColor: '#f8fafc',
                            bodyColor: '#cbd5e1',
                            borderColor: 'rgba(56, 189, 248, 0.2)',
                            borderWidth: 1,
                            padding: 10,
                            displayColors: true,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        if (state.normalizeScale) {
                                            label += context.parsed.y.toFixed(2);
                                        } else {
                                            label += formatCurrency(context.parsed.y);
                                        }
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.04)'
                            },
                            ticks: {
                                color: '#94a3b8',
                                font: {
                                    family: 'Inter',
                                    size: 10
                                }
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.04)'
                            },
                            ticks: {
                                color: '#94a3b8',
                                font: {
                                    family: 'Inter',
                                    size: 10
                                },
                                callback: function(value) {
                                    if (state.normalizeScale) return value;
                                    return formatCurrency(value);
                                }
                            }
                        }
                    }
                }
            });
        }

        // Render Expiration dates selector
        function renderOptionsDropdown(optData) {
            const oldVal = state.optionsDate;
            optionsDateSelect.innerHTML = "";
            
            if (optData.expiration_dates.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "N/A";
                optionsDateSelect.appendChild(opt);
                state.optionsDate = "";
                return;
            }
            
            optData.expiration_dates.forEach(date => {
                const opt = document.createElement("option");
                opt.value = date;
                opt.textContent = date;
                optionsDateSelect.appendChild(opt);
            });
            
            if (optData.expiration_dates.includes(oldVal)) {
                optionsDateSelect.value = oldVal;
                state.optionsDate = oldVal;
            } else {
                optionsDateSelect.value = optData.selected_date;
                state.optionsDate = optData.selected_date;
            }
        }

        // Render side by side options tables
        function renderOptionsTables(optData) {
            callsTableBody.innerHTML = "";
            putsTableBody.innerHTML = "";
            
            if (optData.error) {
                const errRow = `<tr><td colspan="3" style="text-align: center; color: var(--text-secondary);">${optData.error}</td></tr>`;
                callsTableBody.innerHTML = errRow;
                putsTableBody.innerHTML = errRow;
                return;
            }

            if (optData.calls.length === 0 && optData.puts.length === 0) {
                const emptyRow = `<tr><td colspan="3" style="text-align: center; color: var(--text-secondary);">No options data returned.</td></tr>`;
                callsTableBody.innerHTML = emptyRow;
                putsTableBody.innerHTML = emptyRow;
                return;
            }

            // Render Calls
            optData.calls.forEach(item => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 600;">${formatCurrency(item.strike)}</td>
                    <td style="color: var(--accent-green);">${formatCurrency(item.lastPrice)}</td>
                    <td>${item.impliedVolatility.toFixed(2)}%</td>
                `;
                callsTableBody.appendChild(tr);
            });

            // Render Puts
            optData.puts.forEach(item => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 600;">${formatCurrency(item.strike)}</td>
                    <td style="color: var(--accent-red);">${formatCurrency(item.lastPrice)}</td>
                    <td>${item.impliedVolatility.toFixed(2)}%</td>
                `;
                putsTableBody.appendChild(tr);
            });
        }

        // Clear option grids if none available
        function clearOptionsGrids() {
            optionsDateSelect.innerHTML = '<option value="">N/A</option>';
            callsTableBody.innerHTML = '<tr><td colspan="3" class="info-msg">Options data not available for active selection.</td></tr>';
            putsTableBody.innerHTML = '<tr><td colspan="3" class="info-msg">Options data not available for active selection.</td></tr>';
        }

        // Format raw floats into base currency representations
        function formatCurrency(val) {
            if (val === null || isNaN(val)) return "N/A";
            
            const isYen = state.currency === "JPY";
            const decimals = isYen ? 0 : 2;
            
            return state.currencySymbol + val.toLocaleString(undefined, {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            });
        }

        // Helper: HEX color string to RGBA representation (for Chart gradients)
        function hexToRgba(hex, alpha) {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }

        // App Bootstrapping
        window.addEventListener("DOMContentLoaded", () => {
            initUI();
            updateOptionsTickerDropdown();
            triggerTerminalUpdate();
        });
    </script>
    <footer class="terminal-footer">
        © 2026 The Enterprise Equity Terminal Engine • Built by Yusuf Olia
    </footer>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
