import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- BSM CALCULATION FUNCTIONS ---

def calculate_bsm_metrics(S, K, T, r, sigma, q, option_type):
    """
    Calculates the BSM option price and Greeks.
    Allows S to be a scalar or a numpy array for plotting.
    """
    # Handle edge case where T is very close to 0 to avoid division by zero
    T = max(T, 1e-5) 
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        # Theta per day
        theta = (- (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + q * S * np.exp(-q * T) * norm.cdf(d1) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        # Rho per 1% change
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
    else: # Put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        # Theta per day
        theta = (- (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - q * S * np.exp(-q * T) * norm.cdf(-d1) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        # Rho per 1% change
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100

    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    # Vega per 1% change
    vega = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100
    
    return price, delta, gamma, vega, theta, rho


# --- STREAMLIT APP ---

st.set_page_config(page_title="BSM Options Pricing Calculator", layout="wide")
st.title("Black-Scholes-Merton Options Calculator")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Spot price (S)", value=100.0, min_value=0.01, step=1.0)
K = st.sidebar.number_input("Strike price (K)", value=100.0, min_value=0.01, step=1.0)
T = st.sidebar.number_input("Time to expiry (years, T)", value=1.0, min_value=0.00, step=0.1)
r_pct = st.sidebar.number_input("Risk-free rate (r, %)", value=5.0, step=0.1)
sigma_pct = st.sidebar.number_input("Volatility (σ, %)", value=20.0, min_value=0.01, step=1.0)
q_pct = st.sidebar.number_input("Dividend yield (q, %)", value=0.0, step=0.1)
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

# Convert percentages to decimals for calculations
r = r_pct / 100.0
sigma = sigma_pct / 100.0
q = q_pct / 100.0

# --- CALCULATE CURRENT METRICS ---
price, delta, gamma, vega, theta, rho = calculate_bsm_metrics(S, K, T, r, sigma, q, option_type)

# --- DISPLAY METRICS ---
st.subheader(f"Current {option_type} Option Pricing & Greeks")

# Display Price
st.metric(label="Option Price", value=f"${price:.4f}")

# Display Greeks in a clean row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", f"{delta:.4f}")
col2.metric("Gamma", f"{gamma:.4f}")
col3.metric("Vega", f"{vega:.4f}")
col4.metric("Theta", f"{theta:.4f}")
col5.metric("Rho", f"{rho:.4f}")

st.divider()

# --- GENERATE PLOTTING DATA ---
# Range +/- 40% around the current spot price
spot_range = np.linspace(S * 0.6, S * 1.4, 100)
prices, deltas, gammas, vegas, thetas, rhos = calculate_bsm_metrics(
    spot_range, K, T, r, sigma, q, option_type
)

greeks_dict = {
    "Delta": deltas,
    "Gamma": gammas,
    "Vega": vegas,
    "Theta": thetas,
    "Rho": rhos
}

# --- CHARTS ---
st.subheader("Price & Greeks vs. Spot Price")
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Option Price vs Spot Price
with chart_col1:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=spot_range, y=prices, mode='lines', name='Option Price', line=dict(color='blue')))
    fig_price.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current Spot")
    fig_price.update_layout(
        title="Option Price vs. Spot Price",
        xaxis_title="Spot Price",
        yaxis_title="Option Price",
        hovermode="x"
    )
    st.plotly_chart(fig_price, use_container_width=True)

# Chart 2: Selected Greek vs Spot Price
with chart_col2:
    selected_greek = st.selectbox("Select Greek to Plot", options=list(greeks_dict.keys()))
    
    fig_greek = go.Figure()
    fig_greek.add_trace(go.Scatter(x=spot_range, y=greeks_dict[selected_greek], mode='lines', name=selected_greek, line=dict(color='orange')))
    fig_greek.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="Current Spot")
    fig_greek.update_layout(
        title=f"{selected_greek} vs. Spot Price",
        xaxis_title="Spot Price",
        yaxis_title=selected_greek,
        hovermode="x"
    )
    st.plotly_chart(fig_greek, use_container_width=True)
