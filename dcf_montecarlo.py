import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── SETTINGS ──────────────────────────────────────────────
TICKER = input("Enter a stock ticker (e.g. AAPL, TSLA, NVDA): ").upper().strip()
YEARS = 5             # how many years to project
SIMULATIONS = 10000   # number of Monte Carlo runs
WACC = 0.10           # discount rate (10% = typical)
GROWTH_RATE = 0.05    # terminal growth rate (5%)

# ── STEP 1: PULL LIVE DATA ────────────────────────────────
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    current_price = info.get("currentPrice", None)
    shares_outstanding = info.get("sharesOutstanding", None)
    free_cash_flow = info.get("freeCashflow", None)
    revenue_growth = info.get("revenueGrowth", 0.08)

    print(f"\n--- {ticker} ---")
    print(f"Current Price:       ${current_price:,.2f}")
    print(f"Shares Outstanding:  {shares_outstanding:,}")
    print(f"Free Cash Flow:      ${free_cash_flow:,.0f}")
    print(f"Revenue Growth:      {revenue_growth*100:.1f}%")

    return current_price, shares_outstanding, free_cash_flow, revenue_growth

# ── STEP 2: DCF VALUATION ─────────────────────────────────
def run_dcf(fcf, growth_rate, wacc, years, shares):
    projected_fcfs = []
    for year in range(1, years + 1):
        projected_fcf = fcf * (1 + growth_rate) ** year
        discounted_fcf = projected_fcf / (1 + wacc) ** year
        projected_fcfs.append(discounted_fcf)

    # terminal value = value of all cash flows beyond projection period
    terminal_value = (projected_fcfs[-1] * (1 + GROWTH_RATE)) / (wacc - GROWTH_RATE)
    discounted_terminal = terminal_value / (1 + wacc) ** years

    total_value = sum(projected_fcfs) + discounted_terminal
    intrinsic_value_per_share = total_value / shares

    return intrinsic_value_per_share, projected_fcfs

# ── STEP 3: MONTE CARLO SIMULATION ────────────────────────
def run_monte_carlo(fcf, wacc, years, shares, simulations):
    intrinsic_values = []

    for _ in range(simulations):
        # randomise growth rate between 2% and 20%
        rand_growth = np.random.uniform(0.02, 0.20)
        # randomise wacc between 7% and 15%
        rand_wacc = np.random.uniform(0.07, 0.15)
        # randomise terminal growth between 2% and 6%
        rand_terminal = np.random.uniform(0.02, 0.06)

        projected_fcfs = []
        for year in range(1, years + 1):
            projected_fcf = fcf * (1 + rand_growth) ** year
            discounted_fcf = projected_fcf / (1 + rand_wacc) ** year
            projected_fcfs.append(discounted_fcf)

        terminal_value = (projected_fcfs[-1] * (1 + rand_terminal)) / (rand_wacc - rand_terminal)
        discounted_terminal = terminal_value / (1 + rand_wacc) ** years

        total_value = sum(projected_fcfs) + discounted_terminal
        intrinsic_value_per_share = total_value / shares
        intrinsic_values.append(intrinsic_value_per_share)

    return intrinsic_values

# ── STEP 4: VISUALISE ─────────────────────────────────────
def plot_results(intrinsic_values, current_price, dcf_value, ticker):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{ticker} — DCF Valuation + Monte Carlo Simulation", fontsize=14, fontweight="bold")

    # histogram of Monte Carlo outcomes
    ax1.hist(intrinsic_values, bins=100, color="#4A90D9", alpha=0.7, edgecolor="none")
    ax1.axvline(current_price, color="red", linewidth=2, label=f"Current Price: ${current_price:.2f}")
    ax1.axvline(dcf_value, color="green", linewidth=2, linestyle="--", label=f"DCF Value: ${dcf_value:.2f}")
    ax1.axvline(np.percentile(intrinsic_values, 10), color="orange", linewidth=1.5, linestyle=":", label=f"10th percentile: ${np.percentile(intrinsic_values, 10):.2f}")
    ax1.axvline(np.percentile(intrinsic_values, 90), color="purple", linewidth=1.5, linestyle=":", label=f"90th percentile: ${np.percentile(intrinsic_values, 90):.2f}")
    ax1.set_xlabel("Intrinsic Value per Share ($)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Monte Carlo Distribution of Intrinsic Values")
    ax1.legend(fontsize=9)

    # probability chart
    sorted_values = np.sort(intrinsic_values)
    probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax2.plot(sorted_values, probabilities, color="#4A90D9", linewidth=2)
    ax2.axvline(current_price, color="red", linewidth=2, label=f"Current Price: ${current_price:.2f}")
    ax2.axhline(0.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Intrinsic Value per Share ($)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Probability: Stock is Worth at Least $X")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{ticker}_valuation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved as {ticker}_valuation.png")

# ── MAIN ──────────────────────────────────────────────────
def main():
    current_price, shares, fcf, revenue_growth = get_financials(TICKER)

    if not all([current_price, shares, fcf]):
        print("Could not fetch all data. Try a different ticker.")
        return

    # run base DCF
    dcf_value, projected_fcfs = run_dcf(fcf, revenue_growth, WACC, YEARS, shares)
    print(f"\nDCF Intrinsic Value:  ${dcf_value:,.2f}")
    print(f"Current Price:        ${current_price:,.2f}")
    margin = ((dcf_value - current_price) / current_price) * 100
    print(f"Margin of Safety:     {margin:+.1f}%")
    if dcf_value > current_price:
        print("Verdict: UNDERVALUED based on DCF")
    else:
        print("Verdict: OVERVALUED based on DCF")

    # run Monte Carlo
    print(f"\nRunning {SIMULATIONS:,} Monte Carlo simulations...")
    intrinsic_values = run_monte_carlo(fcf, WACC, YEARS, shares, SIMULATIONS)

    print(f"Mean intrinsic value:  ${np.mean(intrinsic_values):,.2f}")
    print(f"Median:                ${np.median(intrinsic_values):,.2f}")
    print(f"10th percentile:       ${np.percentile(intrinsic_values, 10):,.2f}")
    print(f"90th percentile:       ${np.percentile(intrinsic_values, 90):,.2f}")
    prob_undervalued = sum(v > current_price for v in intrinsic_values) / SIMULATIONS * 100
    print(f"Probability stock is undervalued: {prob_undervalued:.1f}%")
    sensitivity_table(fcf, shares, YEARS)
    
    plot_results(intrinsic_values, current_price, dcf_value, TICKER)

# ── STEP 5: SENSITIVITY TABLE ─────────────────────────────
def sensitivity_table(fcf, shares, years):
    wacc_range = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    growth_range = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]

    print("\n--- SENSITIVITY TABLE: Intrinsic Value per Share ---")
    print("WACC →")

    # header row
    header = f"{'Growth ↓':<12}"
    for w in wacc_range:
        header += f"  {w*100:.0f}% WACC"
    print(header)
    print("-" * (12 + len(wacc_range) * 11))

    for g in growth_range:
        row = f"{g*100:.0f}% growth  "
        for w in wacc_range:
            value, _ = run_dcf(fcf, g, w, years, shares)
            row += f"  ${value:>7,.0f}"
        print(row)

    print("\nRead this table like a map:")
    print("Top-right = pessimistic (high WACC, low growth) = lowest value")
    print("Bottom-left = optimistic (low WACC, high growth) = highest value")

if __name__ == "__main__":
    main()