from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
import os

load_dotenv()
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")



# ── SETTINGS ──────────────────────────────────────────────
YEARS = 5
SIMULATIONS = 10000
RISK_FREE_RATE = 0.045   # 10-year US treasury yield ~4.5%
MARKET_RISK_PREMIUM = 0.055  # historical average ~5.5%

# ── STEP 1: PULL LIVE DATA + AUTO WACC ───────────────────
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # core data
    current_price = info.get("currentPrice", None)
    shares = info.get("sharesOutstanding", None)
    fcf = info.get("freeCashflow", None)
    eps = info.get("trailingEps", None)
    reported_growth = info.get("revenueGrowth", 0.05)
    pe_ratio = info.get("trailingPE", None)
    price_to_fcf = info.get("priceToFreeCashflows", None)
    sector = info.get("sector", "Unknown")
    name = info.get("longName", ticker)
    beta = info.get("beta", 1.0)
    total_debt = info.get("totalDebt", 0)
    market_cap = info.get("marketCap", None)
    interest_expense = info.get("interestExpense", None)

    # ── auto WACC calculation
    # cost of equity using CAPM
    cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM

    # cost of debt — use interest expense / total debt if available
    if total_debt and total_debt > 0 and interest_expense and interest_expense < 0:
        cost_of_debt = abs(interest_expense) / total_debt
        cost_of_debt = min(cost_of_debt, 0.12)  # cap at 12%
    else:
        cost_of_debt = 0.04  # default

    # tax rate assumed 21% (US corporate)
    after_tax_cost_of_debt = cost_of_debt * (1 - 0.21)

    # weight of equity vs debt
    if market_cap and total_debt:
        total_capital = market_cap + total_debt
        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital
    else:
        equity_weight = 0.8
        debt_weight = 0.2

    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    wacc = max(0.06, min(wacc, 0.20))  # floor 6%, ceiling 20%

    # ── use EPS as cross-check for FCF
    # if FCF per share is way below EPS, use blend
    fcf_per_share = fcf / shares if fcf and shares else None
    
    if fcf_per_share and eps and eps > 0:
        if fcf_per_share < eps * 0.3:
            # FCF looks suspiciously low — blend with EPS-based estimate
            eps_based_fcf = eps * shares * 0.75  # conservative 75% of earnings
            fcf_adjusted = (fcf * 0.4) + (eps_based_fcf * 0.6)
            fcf_flag = "adjusted (FCF blended with EPS)"
        else:
            fcf_adjusted = fcf
            fcf_flag = "direct FCF"
    else:
        fcf_adjusted = fcf
        fcf_flag = "direct FCF"

    # ── smart growth adjustment
    if reported_growth and reported_growth > 0.5:
        # hypergrowth — decay to sustainable rate
        adjusted_growth = (reported_growth * 0.25) + (0.12 * 0.75)
        growth_flag = "hypergrowth"
        mc_growth_min, mc_growth_max = 0.10, 0.55
    elif reported_growth and reported_growth > 0.2:
        adjusted_growth = (reported_growth * 0.4) + (0.10 * 0.6)
        growth_flag = "high"
        mc_growth_min, mc_growth_max = 0.08, 0.28
    elif reported_growth and reported_growth > 0.05:
        adjusted_growth = reported_growth
        growth_flag = "normal"
        mc_growth_min, mc_growth_max = 0.02, 0.18
    else:
        adjusted_growth = max(reported_growth or 0.03, 0.02)
        growth_flag = "low/stable"
        mc_growth_min, mc_growth_max = 0.01, 0.10

    print(f"\n{'='*58}")
    print(f"  {name} ({ticker})")
    print(f"{'='*58}")
    print(f"  Sector:               {sector}")
    print(f"  Current Price:        ${current_price:,.2f}")
    print(f"  Shares Outstanding:   {shares:,}")
    print(f"  Free Cash Flow:       ${fcf:,.0f}")
    print(f"  FCF per share:        ${fcf_per_share:,.2f}" if fcf_per_share else "  FCF per share:        N/A")
    print(f"  EPS:                  ${eps:,.2f}" if eps else "  EPS:                  N/A")
    print(f"  FCF basis:            {fcf_flag}")
    print(f"  Reported Growth:      {reported_growth*100:.1f}%" if reported_growth else "  Reported Growth:      N/A")
    print(f"  Adjusted Growth:      {adjusted_growth*100:.1f}% ({growth_flag})")
    print(f"\n  ── Auto WACC Calculation ──")
    print(f"  Beta:                 {beta:.2f}")
    print(f"  Cost of Equity:       {cost_of_equity*100:.1f}% (CAPM)")
    print(f"  Cost of Debt:         {cost_of_debt*100:.1f}% (after-tax: {after_tax_cost_of_debt*100:.1f}%)")
    print(f"  Equity Weight:        {equity_weight*100:.0f}%")
    print(f"  Debt Weight:          {debt_weight*100:.0f}%")
    print(f"  AUTO WACC:            {wacc*100:.1f}%")
    print(f"{'='*58}")

    return (current_price, shares, fcf_adjusted, adjusted_growth,
            reported_growth, pe_ratio, price_to_fcf, sector, name,
            wacc, beta, eps, mc_growth_min, mc_growth_max, growth_flag)

# ── STEP 2: DCF ───────────────────────────────────────────
def run_dcf(fcf, growth_rate, wacc, years, shares, terminal_growth=0.03):
    projected_fcfs = []
    for year in range(1, years + 1):
        projected_fcf = fcf * (1 + growth_rate) ** year
        discounted_fcf = projected_fcf / (1 + wacc) ** year
        projected_fcfs.append(discounted_fcf)

    # terminal growth must be below WACC
    terminal_growth = min(terminal_growth, wacc - 0.01)
    terminal_value = (projected_fcfs[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
    discounted_terminal = terminal_value / (1 + wacc) ** years

    total_value = sum(projected_fcfs) + discounted_terminal
    intrinsic_value = total_value / shares
    return intrinsic_value, projected_fcfs

# ── STEP 3: MONTE CARLO ───────────────────────────────────
def run_monte_carlo(fcf, shares, years, simulations, wacc,
                    mc_growth_min, mc_growth_max):
    intrinsic_values = []

    for _ in range(simulations):
        rand_growth = np.random.uniform(mc_growth_min, mc_growth_max)
        # WACC variation ±2% around auto-calculated WACC
        rand_wacc = np.random.uniform(max(0.04, wacc - 0.02), min(0.20, wacc + 0.02))
        rand_terminal = np.random.uniform(0.02, min(0.05, rand_wacc - 0.01))

        projected_fcfs = []
        current_growth = rand_growth
        for year in range(1, years + 1):
            current_growth = current_growth * 0.88
            projected_fcf = fcf * (1 + current_growth) ** year
            discounted_fcf = projected_fcf / (1 + rand_wacc) ** year
            projected_fcfs.append(discounted_fcf)

        if rand_wacc <= rand_terminal:
            continue

        terminal_value = (projected_fcfs[-1] * (1 + rand_terminal)) / (rand_wacc - rand_terminal)
        discounted_terminal = terminal_value / (1 + rand_wacc) ** years
        total_value = sum(projected_fcfs) + discounted_terminal
        intrinsic_value = total_value / shares

        if intrinsic_value > 0:
            intrinsic_values.append(intrinsic_value)

    return intrinsic_values

# ── STEP 4: MULTIPLES ─────────────────────────────────────
def multiples_valuation(current_price, pe_ratio, price_to_fcf, sector, eps):
    sector_pe = {
        "Technology": 28, "Healthcare": 22,
        "Financial Services": 14, "Consumer Cyclical": 20,
        "Industrials": 18, "Energy": 12, "Utilities": 16,
        "Real Estate": 35, "Communication Services": 20,
        "Consumer Defensive": 22, "Basic Materials": 15,
    }
    sector_avg_pe = sector_pe.get(sector, 20)
    estimates = []

    print(f"\n  Multiples Analysis:")
    print(f"  Sector avg P/E:       {sector_avg_pe}x")

    if pe_ratio and pe_ratio > 0 and eps and eps > 0:
        pe_implied = eps * sector_avg_pe
        estimates.append(pe_implied)
        print(f"  Company P/E:          {pe_ratio:.1f}x")
        print(f"  EPS-based value:      ${pe_implied:,.2f}  (EPS × sector avg P/E)")

    if price_to_fcf and price_to_fcf > 0 and pe_ratio:
        sector_avg_pfcf = sector_avg_pe * 0.85
        pfcf_implied = current_price * (sector_avg_pfcf / price_to_fcf)
        estimates.append(pfcf_implied)
        print(f"  Price/FCF implied:    ${pfcf_implied:,.2f}")

    return np.mean(estimates) if estimates else None

# ── STEP 5: BLEND ─────────────────────────────────────────
def blend_valuations(dcf_value, monte_carlo_values, multiples_value, current_price, ticker):
    mc_median = np.median(monte_carlo_values)
    mc_low = np.percentile(monte_carlo_values, 15)
    mc_high = np.percentile(monte_carlo_values, 85)

    print(f"\n{'='*58}")
    print(f"  VALUATION SUMMARY — {ticker}")
    print(f"{'='*58}")
    print(f"  DCF intrinsic value:      ${dcf_value:,.2f}")
    print(f"  Monte Carlo median:       ${mc_median:,.2f}")
    print(f"  Monte Carlo range:        ${mc_low:,.2f} — ${mc_high:,.2f}")
    if multiples_value:
        print(f"  Multiples implied:        ${multiples_value:,.2f}")

    if multiples_value:
        blended = (dcf_value * 0.35) + (mc_median * 0.35) + (multiples_value * 0.30)
        low_target = (dcf_value * 0.35) + (mc_low * 0.35) + (multiples_value * 0.30)
        high_target = (dcf_value * 0.35) + (mc_high * 0.35) + (multiples_value * 0.30)
        weights_used = "35% DCF / 35% Monte Carlo / 30% Multiples"
    else:
        blended = (dcf_value * 0.50) + (mc_median * 0.50)
        low_target = (dcf_value * 0.50) + (mc_low * 0.50)
        high_target = (dcf_value * 0.50) + (mc_high * 0.50)
        weights_used = "50% DCF / 50% Monte Carlo"

    upside = ((blended - current_price) / current_price) * 100
    prob_undervalued = sum(v > current_price for v in monte_carlo_values) / len(monte_carlo_values) * 100

    print(f"\n  Weights:                  {weights_used}")
    print(f"\n  {'─'*48}")
    print(f"  PRICE TARGET:             ${blended:,.2f}")
    print(f"  RANGE:                    ${low_target:,.2f} — ${high_target:,.2f}")
    print(f"  CURRENT PRICE:            ${current_price:,.2f}")
    print(f"  UPSIDE / DOWNSIDE:        {upside:+.1f}%")
    print(f"  PROB. UNDERVALUED:        {prob_undervalued:.1f}%")
    print(f"  {'─'*48}")

    if upside > 20:
        verdict = "BUY — significant upside on fundamentals"
    elif upside > 5:
        verdict = "MILD BUY — modest upside"
    elif upside > -5:
        verdict = "HOLD — fairly valued"
    elif upside > -20:
        verdict = "MILD SELL — modest downside risk"
    else:
        verdict = "SELL — significant overvaluation on fundamentals"

    print(f"  VERDICT:                  {verdict}")
    print(f"{'='*58}")

    return blended, low_target, high_target

# ── STEP 6: NEWS SENTIMENT ────────────────────────────────
# ── STEP 7: NEWS SENTIMENT ────────────────────────────────
def get_news_sentiment(ticker, company_name):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("  No API key found in .env file.")
        return 0, []

    try:
        newsapi = NewsApiClient(api_key=api_key)

        # get a shorter version too — first meaningful word of company name
        # e.g. "Apple Inc." → "Apple", "Coca-Cola Company" → "Coca-Cola"
        company_short = company_name.split()[0].strip(".,")

        # query structure:
        # full name gets highest priority (in quotes = exact match)
        # ticker and short name as fallbacks
        # AND financial keywords to keep results relevant
        query = (
            f'("{company_name}" OR "{ticker}" OR "{company_short}") '
            f'AND (stock OR shares OR earnings OR revenue OR '
            f'investor OR market OR trading OR forecast OR analyst)'
        )

        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=100
        )

        if not articles["articles"]:
            print("  No news articles found.")
            return 0, []

        headlines = []
        scores = []

        print(f"\n  News Sentiment Analysis — last 30 days")
        print(f"  {'─'*48}")

        for article in articles["articles"]:
            headline = article["title"]
            description = article.get("description", "") or ""

            if not headline or headline == "[Removed]":
                continue

            # relevance check — article must mention ticker OR
            # company name OR short name somewhere in headline/description
            combined = (headline + " " + description).lower()
            relevant = (
                ticker.lower() in combined or
                company_name.lower() in combined or
                company_short.lower() in combined
            )

            if not relevant:
                continue

            # score headline weighted more than description
            text_to_score = headline + " " + headline + " " + description
            blob = TextBlob(text_to_score)
            score = blob.sentiment.polarity

            headlines.append({
                "headline": headline[:90],
                "description": description[:120],
                "score": score,
                "source": article["source"]["name"],
            })
            scores.append(score)

            if len(scores) >= 25:
                break

        if not scores:
            print("  No relevant articles found after filtering.")
            print(f"  Query used: {query}")
            return 0, []

        avg_score = np.mean(scores)
        median_score = np.median(scores)
        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - positive - negative

        # blend median and mean — median is more robust to outliers
        final_score = (avg_score * 0.4) + (median_score * 0.6)

        print(f"  Company (full):       {company_name}")
        print(f"  Company (short):      {company_short}")
        print(f"  Articles analysed:    {len(scores)}")
        print(f"  Positive:             {positive}")
        print(f"  Neutral:              {neutral}")
        print(f"  Negative:             {negative}")
        print(f"  Avg sentiment:        {avg_score:+.3f}")
        print(f"  Median sentiment:     {median_score:+.3f}")
        print(f"  Final score:          {final_score:+.3f}  (-1 very negative → +1 very positive)")

        if final_score > 0.25:
            sentiment_label = "STRONGLY BULLISH"
        elif final_score > 0.1:
            sentiment_label = "BULLISH"
        elif final_score > 0.02:
            sentiment_label = "MILDLY BULLISH"
        elif final_score > -0.02:
            sentiment_label = "NEUTRAL"
        elif final_score > -0.1:
            sentiment_label = "MILDLY BEARISH"
        elif final_score > -0.25:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "STRONGLY BEARISH"

        print(f"  Sentiment:            {sentiment_label}")

        print(f"\n  Most impactful headlines:")
        for h in sorted(headlines, key=lambda x: abs(x["score"]), reverse=True)[:5]:
            direction = "BULL" if h["score"] > 0.1 else "BEAR" if h["score"] < -0.1 else "NEUT"
            print(f"  [{direction} {h['score']:+.2f}] {h['headline']}")
            print(f"           Source: {h['source']}")

        return final_score, headlines

    except Exception as e:
        print(f"  News sentiment error: {e}")
        return 0, []    
# ── STEP 7: PRICE PATH SIMULATION ────────────────────────
def price_path_simulation(ticker, current_price, sentiment_score=0, simulations=1000, years=3):
    print(f"\n  Running price path simulation ({years} years)...")

    # pull historical data — 3 years of daily prices
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3y")

    if hist.empty:
        print("  Could not fetch historical data for price simulation.")
        return None, None

    # calculate daily returns and volatility
    daily_returns = hist["Close"].pct_change().dropna()
    mu = daily_returns.mean()        # average daily return
    # adjust mu based on news sentiment — max ±15% adjustment to daily drift
    sentiment_adjustment = sentiment_score * 0.0003
    mu = mu + sentiment_adjustment
    print(f"  Sentiment adjustment: {sentiment_adjustment:+.5f} daily drift")
    sigma = daily_returns.std()      # daily volatility
    trading_days = 252 * years       # trading days in projection period

    print(f"  Daily avg return:     {mu*100:.3f}%")
    print(f"  Daily volatility:     {sigma*100:.3f}%")
    print(f"  Annualised vol:       {sigma * np.sqrt(252) * 100:.1f}%")

    # simulate price paths
    all_paths = []
    for _ in range(simulations):
        prices = [current_price]
        for day in range(trading_days):
            # geometric brownian motion
            shock = np.random.normal(mu, sigma)
            next_price = prices[-1] * (1 + shock)
            next_price = max(next_price, 0.01)  # floor at zero
            prices.append(next_price)
        all_paths.append(prices)

    all_paths = np.array(all_paths)

    # extract year-end prices
    days_per_year = 252
    year_ends = {}
    for y in range(1, years + 1):
        day_idx = min(days_per_year * y, trading_days)
        year_prices = all_paths[:, day_idx]
        year_ends[y] = year_prices

        median = np.median(year_prices)
        low = np.percentile(year_prices, 10)
        high = np.percentile(year_prices, 90)
        prob_up = sum(year_prices > current_price) / simulations * 100

        print(f"\n  Year {y} forecast:")
        print(f"    Median price:       ${median:,.2f}  ({((median-current_price)/current_price)*100:+.1f}%)")
        print(f"    Bear case (10th):   ${low:,.2f}  ({((low-current_price)/current_price)*100:+.1f}%)")
        print(f"    Bull case (90th):   ${high:,.2f}  ({((high-current_price)/current_price)*100:+.1f}%)")
        print(f"    Prob. above today:  {prob_up:.1f}%")

    return all_paths, year_ends

# ── STEP 6: PLOT ──────────────────────────────────────────
def plot_results(ticker, name, current_price, dcf_value, monte_carlo_values,
                 multiples_value, blended, low_target, high_target,
                 wacc, beta, all_paths, year_ends):

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{name} ({ticker}) — Blended Valuation Model v3",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    # ── chart 1: Monte Carlo distribution
    ax1.hist(monte_carlo_values, bins=120, color="#4A90D9",
             alpha=0.65, edgecolor="none", label="MC simulations")
    ax1.axvline(current_price, color="#E74C3C", linewidth=2.5,
                label=f"Current: ${current_price:.2f}")
    ax1.axvline(dcf_value, color="#27AE60", linewidth=2,
                linestyle="--", label=f"DCF: ${dcf_value:.2f}")
    ax1.axvline(blended, color="#F39C12", linewidth=2.5,
                label=f"Price target: ${blended:.2f}")
    ax1.axvspan(low_target, high_target, alpha=0.15, color="#F39C12",
                label=f"Range: ${low_target:.0f}–${high_target:.0f}")
    ax1.set_xlabel("Intrinsic Value per Share ($)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title(f"Monte Carlo Distribution  |  Auto WACC: {wacc*100:.1f}%  |  Beta: {beta:.2f}", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ── chart 2: valuation bar comparison
    labels = ["Current\nPrice", "DCF\nValue", "MC\nMedian", "Price\nTarget"]
    values = [current_price, dcf_value, np.median(monte_carlo_values), blended]
    colors = ["#E74C3C", "#27AE60", "#4A90D9", "#F39C12"]
    if multiples_value:
        labels.insert(3, "Multiples\nImplied")
        values.insert(3, multiples_value)
        colors.insert(3, "#9B59B6")

    bars = ax2.bar(labels, values, color=colors, alpha=0.85,
                   edgecolor="none", width=0.6)
    ax2.axhline(current_price, color="#E74C3C", linewidth=1.5,
                linestyle="--", alpha=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(values) * 0.01,
                 f"${val:.0f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_title("Valuation Methods Compared", fontsize=11)
    ax2.set_ylabel("Value per Share ($)", fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")

    # ── chart 3: cumulative probability
    sorted_vals = np.sort(monte_carlo_values)
    probs = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax3.plot(sorted_vals, probs, color="#4A90D9", linewidth=2.5)
    ax3.axvline(current_price, color="#E74C3C", linewidth=2,
                label=f"Current: ${current_price:.2f}")
    ax3.axvline(blended, color="#F39C12", linewidth=2,
                linestyle="--", label=f"Price target: ${blended:.2f}")
    ax3.axhline(0.5, color="gray", linewidth=1,
                linestyle=":", alpha=0.6, label="50th percentile")
    prob_below = sum(v < current_price for v in monte_carlo_values) / len(monte_carlo_values)
    ax3.axhspan(0, prob_below, alpha=0.08, color="#E74C3C",
                label=f"Overvalued in {prob_below*100:.0f}% of simulations")
    ax3.set_xlabel("Intrinsic Value per Share ($)", fontsize=11)
    ax3.set_ylabel("Cumulative Probability", fontsize=11)
    ax3.set_title("Probability Distribution — Intrinsic Value", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    # ── chart 4: price path simulation
    if all_paths is not None:
        trading_days = all_paths.shape[1]
        x = np.linspace(0, trading_days / 252, trading_days)

        # plot sample paths (faint)
        for path in all_paths[:200]:
            ax4.plot(x, path, color="#4A90D9", alpha=0.04, linewidth=0.5)

        # plot percentile bands
        p10 = np.percentile(all_paths, 10, axis=0)
        p25 = np.percentile(all_paths, 25, axis=0)
        p50 = np.percentile(all_paths, 50, axis=0)
        p75 = np.percentile(all_paths, 75, axis=0)
        p90 = np.percentile(all_paths, 90, axis=0)

        ax4.fill_between(x, p10, p90, alpha=0.15, color="#4A90D9", label="10th–90th percentile")
        ax4.fill_between(x, p25, p75, alpha=0.25, color="#4A90D9", label="25th–75th percentile")
        ax4.plot(x, p50, color="#4A90D9", linewidth=2.5, label="Median path")
        ax4.plot(x, p10, color="#E74C3C", linewidth=1.5, linestyle="--", label="Bear case (10th)")
        ax4.plot(x, p90, color="#27AE60", linewidth=1.5, linestyle="--", label="Bull case (90th)")
        ax4.axhline(current_price, color="#E74C3C", linewidth=1.5,
                    linestyle=":", alpha=0.7, label=f"Today: ${current_price:.2f}")

        # year markers
        for y in range(1, 4):
            ax4.axvline(y, color="gray", linewidth=1, linestyle=":", alpha=0.5)
            if year_ends and y in year_ends:
                median_y = np.median(year_ends[y])
                ax4.annotate(f"Year {y}\n${median_y:.0f}",
                            xy=(y, median_y),
                            xytext=(y + 0.05, median_y * 1.05),
                            fontsize=9, color="#2C3E50",
                            arrowprops=dict(arrowstyle="->", color="gray", lw=1))

        ax4.set_xlabel("Years from Today", fontsize=11)
        ax4.set_ylabel("Simulated Stock Price ($)", fontsize=11)
        ax4.set_title(f"3-Year Price Path Simulation — {ticker}  |  1,000 Monte Carlo paths", fontsize=11)
        ax4.legend(fontsize=9, loc="upper left")
        ax4.grid(True, alpha=0.2)
        ax4.set_xlim(0, 3)

        save = input("\nSave chart as PNG? (y/n): ").strip().lower()
        if save == "y":
            plt.savefig(f"{ticker}_valuation_v3.png", dpi=150, bbox_inches="tight")
            print(f"Chart saved as {ticker}_valuation_v3.png")
    plt.show()

# ── MAIN ──────────────────────────────────────────────────
def main():
    ticker = input("Enter a stock ticker (e.g. AAPL, TSLA, KO): ").upper().strip()

    (current_price, shares, fcf, adjusted_growth, reported_growth,
     pe_ratio, price_to_fcf, sector, name, wacc, beta, eps,
     mc_growth_min, mc_growth_max, growth_flag) = get_financials(ticker)

    if not all([current_price, shares, fcf]):
        print("Could not fetch all required data. Try a different ticker.")
        return

    # DCF
    dcf_value, _ = run_dcf(fcf, adjusted_growth, wacc, YEARS, shares)

    # Monte Carlo
    print(f"\n  Running {SIMULATIONS:,} Monte Carlo simulations...")
    print(f"  Growth range: {mc_growth_min*100:.0f}% — {mc_growth_max*100:.0f}%")
    print(f"  WACC range:   {max(0.04, wacc-0.02)*100:.1f}% — {min(0.20, wacc+0.02)*100:.1f}%")

    monte_carlo_values = run_monte_carlo(
        fcf, shares, YEARS, SIMULATIONS, wacc,
        mc_growth_min, mc_growth_max
    )

    # multiples
    multiples_value = multiples_valuation(
        current_price, pe_ratio, price_to_fcf, sector, eps
    )

    # blend
    blended, low_target, high_target = blend_valuations(
        dcf_value, monte_carlo_values, multiples_value, current_price, ticker
    )

    # plot
    # price path simulation
    # news sentiment
    sentiment_score, headlines = get_news_sentiment(ticker, name)

    # price path simulation — adjusted by sentiment
    all_paths, year_ends = price_path_simulation(
        ticker, current_price, sentiment_score=sentiment_score
    )

    # plot everything
    plot_results(ticker, name, current_price, dcf_value, monte_carlo_values,
                 multiples_value, blended, low_target, high_target,
                 wacc, beta, all_paths, year_ends)
    
if __name__ == "__main__":
    main()