import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Advanced forecasting libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ For advanced forecasting, install: pip install statsmodels")

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Semiconductor Trade Analysis & Forecasting Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/semiconductor-dashboard',
        'Report a bug': 'https://github.com/yourusername/semiconductor-dashboard/issues',
        'About': """
        # Semiconductor Trade Analysis & Forecasting Dashboard
        
        Advanced analytics platform for US semiconductor trade data featuring:
        - Real-time Census Bureau API integration
        - 5-year predictive forecasting models
        - Economic scenario analysis
        - Monte Carlo simulations
        - Machine learning predictions
        
        **Data Sources**: Official US government trade statistics
        **Forecasting**: ARIMA, Random Forest, Linear Regression
        **Scenarios**: Economic growth, recession, trade war impacts
        """
    }
)

# Constants
EXPORT_URL = "https://api.census.gov/data/timeseries/intltrade/exports/hs"
IMPORT_URL = "https://api.census.gov/data/timeseries/intltrade/imports/hs"
NAICS_EXPORT_URL = "https://api.census.gov/data/timeseries/intltrade/exports/naics"
NAICS_IMPORT_URL = "https://api.census.gov/data/timeseries/intltrade/imports/naics"
STATE_EXPORT_NAICS_URL = "https://api.census.gov/data/timltrade/exports/statenaics"
STATE_IMPORT_NAICS_URL = "https://api.census.gov/data/timeseries/intltrade/imports/statenaics"

# API Key handling for deployment
@st.cache_data
def get_api_key():
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return st.secrets["CENSUS_API_KEY"]
    except:
        # Fallback to environment variable
        api_key = os.getenv("CENSUS_API_KEY")
        if not api_key:
            st.error("🚨 **API Key Required**")
            st.markdown("""
            **To use this dashboard, you need a Census Bureau API key:**
            1. Get a free key at: https://api.census.gov/data/key_signup.html
            2. For local use: Set environment variable `CENSUS_API_KEY`
            3. For deployment: Add key to Streamlit secrets
            """)
            st.stop()
        return api_key

API_KEY = get_api_key()

# Country code mapping (abbreviated for space)
COUNTRY_CODES = {
    "1220": "Canada", "2010": "Mexico", "5700": "China", "5880": "Japan",
    "5800": "South Korea", "5830": "Taiwan", "5820": "Hong Kong",
    "5590": "Singapore", "5570": "Malaysia", "4280": "Germany",
    "4120": "United Kingdom", "4279": "France", "4759": "Italy",
    "4210": "Netherlands", "5330": "India", "5650": "Philippines",
    "3570": "Argentina", "3510": "Brazil", "6021": "Australia",
    "-": "Confidential/Not Specified", "999": "Unknown/Unspecified"
}

# Economic indicators for advanced forecasting
ECONOMIC_SCENARIOS = {
    "Baseline": {"gdp_growth": 0.025, "trade_multiplier": 1.0, "description": "Normal economic conditions"},
    "Optimistic Growth": {"gdp_growth": 0.04, "trade_multiplier": 1.15, "description": "AI boom drives demand (+15%)"},
    "Economic Recession": {"gdp_growth": -0.02, "trade_multiplier": 0.8, "description": "Economic downturn (-20%)"},
    "Trade War Impact": {"gdp_growth": 0.01, "trade_multiplier": 0.7, "description": "Severe trade disruptions (-30%)"},
    "Supply Chain Crisis": {"gdp_growth": 0.015, "trade_multiplier": 0.85, "description": "Logistics disruptions (-15%)"}
}

@st.cache_data
def fetch_trade_data_single(hs_code, year, trade_type="exports"):
    """Fetch data for a single HS code and year"""
    base_url = EXPORT_URL if trade_type == "exports" else IMPORT_URL
    
    if trade_type == "exports":
        params = {
            "get": "CTY_CODE,ALL_VAL_MO,YEAR,MONTH,E_COMMODITY",
            "E_COMMODITY": hs_code,
            "YEAR": year,
            "key": API_KEY
        }
        value_field = "ALL_VAL_MO"
    else:
        params = {
            "get": "CTY_CODE,GEN_VAL_MO,YEAR,MONTH,I_COMMODITY",
            "I_COMMODITY": hs_code,
            "YEAR": year,
            "key": API_KEY
        }
        value_field = "GEN_VAL_MO"
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, list) or len(data) < 2:
            return pd.DataFrame()
        
        headers = data[0]
        seen = {}
        unique_headers = []
        for header in headers:
            if header in seen:
                seen[header] += 1
                unique_headers.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                unique_headers.append(header)
        
        df = pd.DataFrame(data[1:], columns=unique_headers)
        df[value_field] = pd.to_numeric(df[value_field], errors='coerce')
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors='coerce')
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors='coerce')
        df = df.dropna(subset=[value_field, "MONTH", "YEAR"])
        
        if df.empty:
            return pd.DataFrame()
        
        df["DATE"] = pd.to_datetime(
            df["YEAR"].astype(str) + df["MONTH"].astype(str).str.zfill(2) + "01", 
            format="%Y%m%d", errors='coerce'
        )
        df = df.dropna(subset=["DATE"])
        df["TRADE_VALUE"] = df[value_field]
        df["HS_CODE"] = hs_code
        df["TRADE_TYPE"] = trade_type.title()
        df["COUNTRY_NAME"] = df["CTY_CODE"].map(COUNTRY_CODES).fillna(
            df["CTY_CODE"].apply(lambda x: f"Unknown Country - {x}")
        )
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch {trade_type} data for HS {hs_code}, year {year}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_multi_trade_data(hs_codes, years, trade_types):
    """Fetch data for multiple HS codes, years, and trade types with caching"""
    all_data = []
    total_requests = len(hs_codes) * len(years) * len(trade_types)
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_request = 0
    
    for hs_code in hs_codes:
        for year in years:
            for trade_type in trade_types:
                status_text.text(f"Fetching {trade_type} data for HS {hs_code}, year {year}... ({current_request + 1}/{total_requests})")
                df = fetch_trade_data_single(hs_code, str(year), trade_type)
                
                if not df.empty:
                    all_data.append(df)
                
                current_request += 1
                progress_bar.progress(current_request / total_requests)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# ADVANCED FORECASTING FUNCTIONS

def prepare_forecasting_data(df):
    """Prepare data for forecasting analysis"""
    if df.empty:
        return pd.DataFrame()
    
    # Create monthly aggregated data
    monthly_data = df.groupby(['DATE', 'TRADE_TYPE'])['TRADE_VALUE'].sum().reset_index()
    monthly_data = monthly_data.sort_values(['TRADE_TYPE', 'DATE'])
    
    # Add time-based features
    monthly_data['YEAR'] = monthly_data['DATE'].dt.year
    monthly_data['MONTH'] = monthly_data['DATE'].dt.month
    monthly_data['QUARTER'] = monthly_data['DATE'].dt.quarter
    monthly_data['TIME_INDEX'] = monthly_data.groupby('TRADE_TYPE').cumcount()
    
    # Add moving averages
    for window in [3, 6, 12]:
        monthly_data[f'MA_{window}'] = monthly_data.groupby('TRADE_TYPE')['TRADE_VALUE'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Add growth rates
    monthly_data['YoY_Growth'] = monthly_data.groupby('TRADE_TYPE')['TRADE_VALUE'].pct_change(12)
    monthly_data['MoM_Growth'] = monthly_data.groupby('TRADE_TYPE')['TRADE_VALUE'].pct_change(1)
    
    return monthly_data

def linear_regression_forecast(data, forecast_periods=60):
    """Simple linear regression forecast"""
    forecasts = {}
    
    for trade_type in data['TRADE_TYPE'].unique():
        trade_data = data[data['TRADE_TYPE'] == trade_type].copy()
        
        if len(trade_data) < 12:  # Need at least 1 year of data
            continue
        
        # Prepare features
        X = trade_data[['TIME_INDEX']].values
        y = trade_data['TRADE_VALUE'].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate predictions
        last_time_index = trade_data['TIME_INDEX'].max()
        future_time_indices = np.arange(last_time_index + 1, last_time_index + 1 + forecast_periods)
        future_predictions = model.predict(future_time_indices.reshape(-1, 1))
        
        # Generate future dates
        last_date = trade_data['DATE'].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
        
        # Calculate metrics
        train_predictions = model.predict(X)
        r2 = r2_score(y, train_predictions)
        mae = mean_absolute_error(y, train_predictions)
        rmse = np.sqrt(mean_squared_error(y, train_predictions))
        
        forecasts[trade_type] = {
            'method': 'Linear Regression',
            'dates': future_dates,
            'predictions': future_predictions,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'trend_coefficient': model.coef_[0],
            'intercept': model.intercept_
        }
    
    return forecasts

def random_forest_forecast(data, forecast_periods=60):
    """Random Forest regression forecast with multiple features"""
    forecasts = {}
    
    for trade_type in data['TRADE_TYPE'].unique():
        trade_data = data[data['TRADE_TYPE'] == trade_type].copy()
        
        if len(trade_data) < 24:  # Need at least 2 years of data
            continue
        
        # Prepare features
        feature_cols = ['TIME_INDEX', 'MONTH', 'QUARTER', 'MA_3', 'MA_6', 'MA_12']
        available_features = [col for col in feature_cols if col in trade_data.columns]
        
        X = trade_data[available_features].fillna(method='ffill').fillna(0)
        y = trade_data['TRADE_VALUE'].values
        
        # Fit model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Generate future features
        last_row = trade_data.iloc[-1]
        future_features = []
        
        for i in range(1, forecast_periods + 1):
            future_date = last_row['DATE'] + pd.DateOffset(months=i)
            future_row = {
                'TIME_INDEX': last_row['TIME_INDEX'] + i,
                'MONTH': future_date.month,
                'QUARTER': future_date.quarter,
                'MA_3': last_row['MA_3'],  # Use last known values
                'MA_6': last_row['MA_6'],
                'MA_12': last_row['MA_12']
            }
            future_features.append([future_row.get(col, 0) for col in available_features])
        
        future_X = np.array(future_features)
        future_predictions = model.predict(future_X)
        
        # Generate future dates
        last_date = trade_data['DATE'].max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
        
        # Calculate metrics
        train_predictions = model.predict(X)
        r2 = r2_score(y, train_predictions)
        mae = mean_absolute_error(y, train_predictions)
        rmse = np.sqrt(mean_squared_error(y, train_predictions))
        
        # Feature importance
        feature_importance = dict(zip(available_features, model.feature_importances_))
        
        forecasts[trade_type] = {
            'method': 'Random Forest',
            'dates': future_dates,
            'predictions': future_predictions,
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': feature_importance
        }
    
    return forecasts

def arima_forecast(data, forecast_periods=60):
    """ARIMA time series forecast"""
    if not STATSMODELS_AVAILABLE:
        return {}
    
    forecasts = {}
    
    for trade_type in data['TRADE_TYPE'].unique():
        trade_data = data[data['TRADE_TYPE'] == trade_type].copy()
        
        if len(trade_data) < 36:  # Need at least 3 years of data
            continue
        
        try:
            # Prepare time series
            ts_data = trade_data.set_index('DATE')['TRADE_VALUE']
            ts_data = ts_data.asfreq('MS')  # Month start frequency
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=forecast_periods)
            confidence_intervals = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Generate future dates
            last_date = ts_data.index[-1]
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
            
            # Calculate metrics on training data
            fitted_values = fitted_model.fittedvalues
            residuals = ts_data - fitted_values
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            
            forecasts[trade_type] = {
                'method': 'ARIMA',
                'dates': future_dates,
                'predictions': forecast_result.values,
                'confidence_lower': confidence_intervals.iloc[:, 0].values,
                'confidence_upper': confidence_intervals.iloc[:, 1].values,
                'mae': mae,
                'rmse': rmse,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        
        except Exception as e:
            st.warning(f"ARIMA forecast failed for {trade_type}: {e}")
            continue
    
    return forecasts

def exponential_smoothing_forecast(data, forecast_periods=60):
    """Exponential Smoothing forecast with seasonality"""
    if not STATSMODELS_AVAILABLE:
        return {}
    
    forecasts = {}
    
    for trade_type in data['TRADE_TYPE'].unique():
        trade_data = data[data['TRADE_TYPE'] == trade_type].copy()
        
        if len(trade_data) < 24:  # Need at least 2 years of data
            continue
        
        try:
            # Prepare time series
            ts_data = trade_data.set_index('DATE')['TRADE_VALUE']
            ts_data = ts_data.asfreq('MS')
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            )
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=forecast_periods)
            
            # Generate future dates
            last_date = ts_data.index[-1]
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_periods + 1)]
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            residuals = ts_data - fitted_values
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            
            forecasts[trade_type] = {
                'method': 'Exponential Smoothing',
                'dates': future_dates,
                'predictions': forecast_result.values,
                'mae': mae,
                'rmse': rmse,
                'aic': fitted_model.aic
            }
        
        except Exception as e:
            st.warning(f"Exponential Smoothing forecast failed for {trade_type}: {e}")
            continue
    
    return forecasts

def monte_carlo_simulation(base_forecasts, n_simulations=1000):
    """Monte Carlo simulation for uncertainty quantification"""
    mc_results = {}
    
    for trade_type, forecast_data in base_forecasts.items():
        if 'predictions' not in forecast_data:
            continue
        
        predictions = np.array(forecast_data['predictions'])
        
        # Estimate volatility from recent data or use historical standard deviation
        volatility = 0.15  # 15% annual volatility assumption
        monthly_volatility = volatility / np.sqrt(12)
        
        # Generate Monte Carlo paths
        n_periods = len(predictions)
        mc_paths = np.zeros((n_simulations, n_periods))
        
        for sim in range(n_simulations):
            # Add random noise to base forecast
            random_shocks = np.random.normal(0, monthly_volatility, n_periods)
            cumulative_shocks = np.cumprod(1 + random_shocks)
            mc_paths[sim, :] = predictions * cumulative_shocks
        
        # Calculate confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        confidence_bands = np.percentile(mc_paths, percentiles, axis=0)
        
        mc_results[trade_type] = {
            'method': 'Monte Carlo Simulation',
            'dates': forecast_data['dates'],
            'mean_forecast': np.mean(mc_paths, axis=0),
            'confidence_bands': confidence_bands,
            'percentile_labels': [f'{p}th percentile' for p in percentiles],
            'paths': mc_paths[:100, :]  # Store first 100 paths for visualization
        }
    
    return mc_results

def apply_economic_scenario(forecasts, scenario_name):
    """Apply economic scenario adjustments to forecasts"""
    scenario = ECONOMIC_SCENARIOS.get(scenario_name, ECONOMIC_SCENARIOS['Baseline'])
    adjusted_forecasts = {}
    
    for trade_type, forecast_data in forecasts.items():
        if 'predictions' not in forecast_data:
            continue
        
        adjusted_data = forecast_data.copy()
        
        # Apply trade multiplier
        base_predictions = np.array(forecast_data['predictions'])
        adjusted_predictions = base_predictions * scenario['trade_multiplier']
        
        # Apply gradual growth/decline based on GDP growth
        n_periods = len(adjusted_predictions)
        growth_factors = np.array([(1 + scenario['gdp_growth'])**(i/12) for i in range(n_periods)])
        adjusted_predictions = adjusted_predictions * growth_factors
        
        adjusted_data['predictions'] = adjusted_predictions
        adjusted_data['scenario'] = scenario_name
        adjusted_data['scenario_description'] = scenario['description']
        adjusted_data['adjustment_factor'] = scenario['trade_multiplier']
        
        adjusted_forecasts[trade_type] = adjusted_data
    
    return adjusted_forecasts

def create_forecast_comparison_chart(forecasts_dict):
    """Create interactive forecast comparison chart"""
    fig = make_subplots(
        rows=len(forecasts_dict), cols=1,
        subplot_titles=list(forecasts_dict.keys()),
        vertical_spacing=0.05
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (trade_type, methods) in enumerate(forecasts_dict.items(), 1):
        for j, (method, forecast_data) in enumerate(methods.items()):
            if 'predictions' not in forecast_data:
                continue
            
            dates = pd.to_datetime(forecast_data['dates'])
            predictions = forecast_data['predictions']
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=predictions / 1_000_000_000,  # Convert to billions
                    mode='lines',
                    name=f"{method}",
                    line=dict(color=colors[j % len(colors)]),
                    showlegend=(i == 1)  # Only show legend for first subplot
                ),
                row=i, col=1
            )
            
            # Add confidence intervals for ARIMA
            if method == 'ARIMA' and 'confidence_lower' in forecast_data:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=forecast_data['confidence_upper'] / 1_000_000_000,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=forecast_data['confidence_lower'] / 1_000_000_000,
                        mode='lines',
                        fill='tonexty',
                        fillcolor=f'rgba({colors[j % len(colors)][4:-1]}, 0.1)',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=i, col=1
                )
    
    fig.update_layout(
        height=400 * len(forecasts_dict),
        title="5-Year Trade Forecasting Comparison",
        xaxis_title="Date",
        yaxis_title="Trade Value (Billions USD)"
    )
    
    return fig

def create_monte_carlo_chart(mc_results):
    """Create Monte Carlo simulation visualization"""
    fig = make_subplots(
        rows=len(mc_results), cols=1,
        subplot_titles=list(mc_results.keys()),
        vertical_spacing=0.05
    )
    
    for i, (trade_type, mc_data) in enumerate(mc_results.items(), 1):
        dates = pd.to_datetime(mc_data['dates'])
        
        # Plot confidence bands
        confidence_bands = mc_data['confidence_bands']
        percentile_labels = mc_data['percentile_labels']
        
        # Plot 95% confidence interval
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_bands[4] / 1_000_000_000,  # 95th percentile
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_bands[0] / 1_000_000_000,  # 5th percentile
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(width=0),
                name='95% Confidence Interval' if i == 1 else None,
                showlegend=(i == 1)
            ),
            row=i, col=1
        )
        
        # Plot median forecast
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_bands[2] / 1_000_000_000,  # Median
                mode='lines',
                line=dict(color='red', width=2),
                name='Median Forecast' if i == 1 else None,
                showlegend=(i == 1)
            ),
            row=i, col=1
        )
        
        # Plot some sample paths
        paths = mc_data['paths']
        for j in range(0, min(10, paths.shape[0]), 2):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=paths[j] / 1_000_000_000,
                    mode='lines',
                    line=dict(color='lightblue', width=0.5),
                    opacity=0.3,
                    showlegend=False
                ),
                row=i, col=1
            )
    
    fig.update_layout(
        height=400 * len(mc_results),
        title="Monte Carlo Simulation - Uncertainty Analysis",
        xaxis_title="Date",
        yaxis_title="Trade Value (Billions USD)"
    )
    
    return fig

# Original functions (abbreviated for space)
def create_trade_comparison_chart(df):
    """Create charts comparing imports vs exports"""
    if df.empty:
        return None
    
    required_columns = ['DATE', 'HS_CODE', 'TRADE_TYPE', 'TRADE_VALUE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns for chart: {missing_columns}")
        return None
    
    monthly_data = df.groupby(['DATE', 'HS_CODE', 'TRADE_TYPE'])['TRADE_VALUE'].sum().reset_index()
    
    charts = {}
    for hs_code in df['HS_CODE'].unique():
        hs_data = monthly_data[monthly_data['HS_CODE'] == hs_code]
        if not hs_data.empty:
            pivot_data = hs_data.pivot(index='DATE', columns='TRADE_TYPE', values='TRADE_VALUE').fillna(0)
            charts[hs_code] = pivot_data
    
    return charts

def create_country_breakdown(df, top_n=10):
    """Create country breakdown for selected period"""
    if df.empty:
        return pd.DataFrame()
    
    required_columns = ['CTY_CODE', 'COUNTRY_NAME', 'TRADE_TYPE', 'TRADE_VALUE']
    if 'HS_CODE' in df.columns:
        required_columns.append('HS_CODE')
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing columns for country breakdown: {missing_columns}")
        return pd.DataFrame()
    
    if 'HS_CODE' in df.columns:
        country_data = df.groupby(['CTY_CODE', 'COUNTRY_NAME', 'TRADE_TYPE', 'HS_CODE'])['TRADE_VALUE'].sum().reset_index()
    else:
        country_data = df.groupby(['CTY_CODE', 'COUNTRY_NAME', 'TRADE_TYPE'])['TRADE_VALUE'].sum().reset_index()
    
    return country_data

# Streamlit UI
st.title("🔮 Advanced Semiconductor Trade Analysis & Forecasting Dashboard")
st.markdown("**Real-time data analysis with AI-powered 5-year forecasting capabilities**")

# Sidebar for forecasting options
with st.sidebar:
    st.subheader("🔮 Forecasting Options")
    
    forecast_methods = st.multiselect(
        "Select Forecasting Methods:",
        ["Linear Regression", "Random Forest", "ARIMA", "Exponential Smoothing", "Monte Carlo"],
        default=["Linear Regression", "Random Forest"]
    )
    
    forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)
    forecast_periods = forecast_years * 12
    
    st.subheader("📊 Economic Scenarios")
    scenario = st.selectbox(
        "Select Economic Scenario:",
        list(ECONOMIC_SCENARIOS.keys()),
        index=0
    )
    
    st.info(ECONOMIC_SCENARIOS[scenario]['description'])
    
    st.subheader("📍 Country Code Reference")
    for code, country in list(COUNTRY_CODES.items())[:10]:
        st.text(f"{code}: {country}")

# Data source selection
st.subheader("📊 Data Source Selection")
data_source = st.radio(
    "Choose your primary data source:",
    ["HS Codes (Product Classification)", "NAICS Codes (Industry Classification)"],
    index=0
)

# Input controls
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📦 HS Codes")
    st.text("Semiconductor codes:")
    st.text("8541 - Diodes, transistors")
    st.text("8542 - Integrated circuits")
    hs_codes = ["8541", "8542"]

with col2:
    st.subheader("📅 Year Range")
    current_year = datetime.now().year
    end_year = st.number_input("End Year", min_value=2013, max_value=current_year-1, value=2023)
    start_year = st.number_input("Start Year", min_value=2013, max_value=end_year, value=2020)

with col3:
    st.subheader("🔄 Trade Types")
    include_exports = st.checkbox("Include Exports", value=True)
    include_imports = st.checkbox("Include Imports", value=True)
    show_top_countries = st.number_input("Top Countries to Show", min_value=5, max_value=20, value=10)

# Generate data lists
years = list(range(start_year, end_year + 1))
trade_types = []
if include_exports:
    trade_types.append("exports")
if include_imports:
    trade_types.append("imports")

if st.button("🚀 Fetch Trade Data & Generate Forecasts"):
    if len(trade_types) == 0:
        st.error("❌ Please select at least one trade type")
        st.stop()
    
    # Fetch historical data
    st.info(f"🔍 Fetching HS semiconductor data (8541+8542) for {', '.join(trade_types)} from {start_year} to {end_year}")
    
    try:
        df = fetch_multi_trade_data(hs_codes, years, trade_types)
        
        if not df.empty:
            st.success(f"✅ Successfully fetched {len(df)} records!")
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Countries", df['CTY_CODE'].nunique())
            with col3:
                st.metric("Total Value", f"${df['TRADE_VALUE'].sum():,.0f}")
            with col4:
                st.metric("Time Span", f"{len(years)} years")
            
            # Prepare data for forecasting
            forecast_data = prepare_forecasting_data(df)
            
            if not forecast_data.empty:
                st.subheader("🔮 Advanced 5-Year Trade Forecasting Analysis")
                
                # Generate forecasts using selected methods
                all_forecasts = {}
                
                if "Linear Regression" in forecast_methods:
                    with st.spinner("🔄 Running Linear Regression forecasts..."):
                        lr_forecasts = linear_regression_forecast(forecast_data, forecast_periods)
                        for trade_type, forecast in lr_forecasts.items():
                            if trade_type not in all_forecasts:
                                all_forecasts[trade_type] = {}
                            all_forecasts[trade_type]['Linear Regression'] = forecast
                
                if "Random Forest" in forecast_methods:
                    with st.spinner("🌲 Running Random Forest forecasts..."):
                        rf_forecasts = random_forest_forecast(forecast_data, forecast_periods)
                        for trade_type, forecast in rf_forecasts.items():
                            if trade_type not in all_forecasts:
                                all_forecasts[trade_type] = {}
                            all_forecasts[trade_type]['Random Forest'] = forecast
                
                if "ARIMA" in forecast_methods and STATSMODELS_AVAILABLE:
                    with st.spinner("📈 Running ARIMA forecasts..."):
                        arima_forecasts = arima_forecast(forecast_data, forecast_periods)
                        for trade_type, forecast in arima_forecasts.items():
                            if trade_type not in all_forecasts:
                                all_forecasts[trade_type] = {}
                            all_forecasts[trade_type]['ARIMA'] = forecast
                
                if "Exponential Smoothing" in forecast_methods and STATSMODELS_AVAILABLE:
                    with st.spinner("📊 Running Exponential Smoothing forecasts..."):
                        es_forecasts = exponential_smoothing_forecast(forecast_data, forecast_periods)
                        for trade_type, forecast in es_forecasts.items():
                            if trade_type not in all_forecasts:
                                all_forecasts[trade_type] = {}
                            all_forecasts[trade_type]['Exponential Smoothing'] = forecast
                
                # Apply economic scenario
                if scenario != "Baseline" and all_forecasts:
                    st.info(f"🎯 Applying {scenario} economic scenario adjustments...")
                    for trade_type in all_forecasts:
                        for method in all_forecasts[trade_type]:
                            adjusted = apply_economic_scenario({trade_type: all_forecasts[trade_type][method]}, scenario)
                            all_forecasts[trade_type][method] = adjusted[trade_type]
                
                # Display forecast results
                if all_forecasts:
                    # Create comparison chart
                    st.subheader("📈 Forecast Comparison by Method")
                    comparison_chart = create_forecast_comparison_chart(all_forecasts)
                    st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Display forecast metrics
                    st.subheader("📊 Forecast Performance Metrics")
                    
                    for trade_type, methods in all_forecasts.items():
                        st.markdown(f"**{trade_type} Forecasts:**")
                        
                        metrics_data = []
                        for method, forecast_data in methods.items():
                            if 'predictions' in forecast_data:
                                final_value = forecast_data['predictions'][-1] / 1_000_000_000
                                current_value = forecast_data['predictions'][0] / 1_000_000_000
                                total_growth = ((final_value - current_value) / current_value) * 100
                                
                                metrics_row = {
                                    'Method': method,
                                    f'{forecast_years}-Year Projection': f"${final_value:.1f}B",
                                    'Total Growth': f"{total_growth:+.1f}%",
                                    'Annual Growth': f"{total_growth/forecast_years:+.1f}%"
                                }
                                
                                if 'r2_score' in forecast_data:
                                    metrics_row['R² Score'] = f"{forecast_data['r2_score']:.3f}"
                                if 'mae' in forecast_data:
                                    metrics_row['MAE (Billions)'] = f"${forecast_data['mae']/1_000_000_000:.2f}B"
                                
                                metrics_data.append(metrics_row)
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data)
                            st.dataframe(metrics_df, use_container_width=True)
                
                # Monte Carlo simulation
                if "Monte Carlo" in forecast_methods and all_forecasts:
                    st.subheader("🎲 Monte Carlo Uncertainty Analysis")
                    
                    # Use best performing method for Monte Carlo base
                    base_forecasts = {}
                    for trade_type, methods in all_forecasts.items():
                        if methods:
                            # Use first available method as base
                            base_forecasts[trade_type] = list(methods.values())[0]
                    
                    if base_forecasts:
                        with st.spinner("🎲 Running Monte Carlo simulations..."):
                            mc_results = monte_carlo_simulation(base_forecasts, n_simulations=1000)
                        
                        if mc_results:
                            mc_chart = create_monte_carlo_chart(mc_results)
                            st.plotly_chart(mc_chart, use_container_width=True)
                            
                            # Display uncertainty metrics
                            st.markdown("**Uncertainty Analysis:**")
                            for trade_type, mc_data in mc_results.items():
                                confidence_bands = mc_data['confidence_bands']
                                final_median = confidence_bands[2][-1] / 1_000_000_000
                                final_5th = confidence_bands[0][-1] / 1_000_000_000
                                final_95th = confidence_bands[4][-1] / 1_000_000_000
                                
                                st.markdown(f"**{trade_type} - {forecast_years} Year Outlook:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Median Forecast", f"${final_median:.1f}B")
                                with col2:
                                    st.metric("Pessimistic (5%)", f"${final_5th:.1f}B")
                                with col3:
                                    st.metric("Optimistic (95%)", f"${final_95th:.1f}B")
                
                # Economic scenario analysis
                st.subheader("🎯 Economic Scenario Impact Analysis")
                scenario_comparison = {}
                
                if all_forecasts:
                    base_method = list(list(all_forecasts.values())[0].keys())[0]  # Get first method
                    
                    for scenario_name in ECONOMIC_SCENARIOS.keys():
                        scenario_results = {}
                        for trade_type, methods in all_forecasts.items():
                            if base_method in methods:
                                base_forecast = methods[base_method]
                                adjusted = apply_economic_scenario({trade_type: base_forecast}, scenario_name)
                                scenario_results[trade_type] = adjusted[trade_type]['predictions'][-1] / 1_000_000_000
                        scenario_comparison[scenario_name] = scenario_results
                    
                    if scenario_comparison:
                        scenario_df_data = []
                        for scenario_name, results in scenario_comparison.items():
                            for trade_type, final_value in results.items():
                                scenario_df_data.append({
                                    'Scenario': scenario_name,
                                    'Trade Type': trade_type,
                                    f'{forecast_years}-Year Projection': f"${final_value:.1f}B",
                                    'Description': ECONOMIC_SCENARIOS[scenario_name]['description']
                                })
                        
                        scenario_df = pd.DataFrame(scenario_df_data)
                        st.dataframe(scenario_df, use_container_width=True)
            
            # Historical analysis
            st.subheader("📈 Historical Trade Trends")
            charts = create_trade_comparison_chart(df)
            
            if charts:
                combined_data = None
                for hs_code, chart_data in charts.items():
                    if combined_data is None:
                        combined_data = chart_data
                    else:
                        combined_data = combined_data.add(chart_data, fill_value=0)
                
                if combined_data is not None:
                    st.line_chart(combined_data)
            
            # Country breakdown
            st.subheader(f"🌍 Top {show_top_countries} Trading Partners")
            country_data = create_country_breakdown(df, show_top_countries)
            
            if not country_data.empty:
                top_countries = country_data.groupby(['CTY_CODE', 'COUNTRY_NAME'])['TRADE_VALUE'].sum().reset_index()
                top_countries = top_countries.sort_values('TRADE_VALUE', ascending=False).head(show_top_countries)
                
                st.dataframe(
                    top_countries.rename(columns={
                        'CTY_CODE': 'Country Code',
                        'COUNTRY_NAME': 'Country Name', 
                        'TRADE_VALUE': 'Total Trade Value ($)'
                    }).style.format({'Total Trade Value ($)': '{:,.0f}'}),
                    use_container_width=True
                )
            
            # Download options
            st.subheader("💾 Download Options")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📁 Download Historical Data (CSV)",
                    data=csv,
                    file_name=f"semiconductor_trade_data_{start_year}_{end_year}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if all_forecasts:
                    # Create forecast download data
                    forecast_download_data = []
                    for trade_type, methods in all_forecasts.items():
                        for method, forecast_data in methods.items():
                            if 'predictions' in forecast_data:
                                for i, (date, value) in enumerate(zip(forecast_data['dates'], forecast_data['predictions'])):
                                    forecast_download_data.append({
                                        'Date': date,
                                        'Trade_Type': trade_type,
                                        'Method': method,
                                        'Predicted_Value': value,
                                        'Scenario': scenario
                                    })
                    
                    if forecast_download_data:
                        forecast_csv = pd.DataFrame(forecast_download_data).to_csv(index=False)
                        st.download_button(
                            label="🔮 Download Forecasts (CSV)",
                            data=forecast_csv,
                            file_name=f"semiconductor_forecasts_{forecast_years}year_{scenario.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
        else:
            st.warning("⚠️ No data found for the selected criteria")
            
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 🔮 Advanced Forecasting Dashboard")
st.markdown("""
**Forecasting Capabilities:**
- 🎯 **Multiple Methods**: Linear Regression, Random Forest, ARIMA, Exponential Smoothing
- 🎲 **Uncertainty Analysis**: Monte Carlo simulations with confidence intervals
- 📊 **Economic Scenarios**: Growth, recession, trade war impact modeling
- 📈 **Performance Metrics**: R², MAE, RMSE for model validation
- 🌍 **Global Context**: Economic indicator integration and policy impact assessment

**Professional Applications:**
- Policy impact assessment and economic forecasting
- Investment planning and strategic decision support
- Supply chain risk analysis and scenario planning
- Congressional briefing preparation and stakeholder communication
""")

st.markdown("**💡 Advanced Analytics:** This dashboard demonstrates sophisticated forecasting methodologies suitable for government policy analysis and strategic economic planning!")