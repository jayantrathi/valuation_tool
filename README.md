# Stock Valuation Tool

A Python-based stock valuation tool that combines multiple methodologies 
to produce a blended price target and 3-year price forecast for any stock.

## What it does

- **DCF Valuation** — projects free cash flows forward and discounts them 
  back using an automatically calculated WACC
- **Monte Carlo Simulation** — runs 10,000 simulations with randomised 
  growth and discount rate assumptions to model valuation uncertainty
- **Multiples Analysis** — compares the company's P/E and Price/FCF 
  ratios against sector averages to produce a multiples-implied value
- **Blended Price Target** — weights all three methods into a single 
  price target with a confidence range
- **News Sentiment Analysis** — pulls the last 30 days of relevant news 
  headlines via NewsAPI, scores them using TextBlob sentiment analysis, 
  and incorporates the result as a drift adjustment in the price simulation
- **3-Year Price Path Simulation** — uses geometric Brownian motion and 
  3 years of historical price data to simulate 1,000 possible price paths, 
  with year 1, 2, and 3 median, bear, and bull case forecasts

## Example output

Running on Apple Inc. (AAPL):
==========================================================
  Apple Inc. (AAPL)
==========================================================
  Sector:               Technology
  Current Price:        $283.78
  Shares Outstanding:   14,687,356,000
  Free Cash Flow:       $101,090,746,368
  FCF per share:        $6.88
  EPS:                  $8.26
  FCF basis:            direct FCF
  Reported Growth:      16.6%
  Adjusted Growth:      16.6% (normal)

  ── Auto WACC Calculation ──
  Beta:                 1.09
  Cost of Equity:       10.5% (CAPM)
  Cost of Debt:         4.0% (after-tax: 3.2%)
  Equity Weight:        98%
  Debt Weight:          2%
  AUTO WACC:            10.3%
==========================================================

  Running 10,000 Monte Carlo simulations...
  Growth range: 2% — 18%
  WACC range:   8.3% — 12.3%

  Multiples Analysis:
  Sector avg P/E:       28x
  Company P/E:          34.4x
  EPS-based value:      $231.28  (EPS × sector avg P/E)

==========================================================
  VALUATION SUMMARY — AAPL
==========================================================
  DCF intrinsic value:      $118.79
  Monte Carlo median:       $82.17
  Monte Carlo range:        $65.53 — $108.16
  Multiples implied:        $231.28

  Weights:                  35% DCF / 35% Monte Carlo / 30% Multiples

  ────────────────────────────────────────────────
  PRICE TARGET:             $139.72
  RANGE:                    $133.90 — $148.82
  CURRENT PRICE:            $283.78
  UPSIDE / DOWNSIDE:        -50.8%
  PROB. UNDERVALUED:        0.0%
  ────────────────────────────────────────────────
  VERDICT:                  SELL — significant overvaluation on fundamentals
==========================================================

  News Sentiment Analysis — last 30 days
  ────────────────────────────────────────────────
  Company (full):       Apple Inc.
  Company (short):      Apple
  Articles analysed:    25
  Positive:             15
  Neutral:              8
  Negative:             2
  Avg sentiment:        +0.170
  Median sentiment:     +0.166
  Final score:          +0.168  (-1 very negative → +1 very positive)
  Sentiment:            BULLISH

  Most impactful headlines:
  [BULL +0.70] Tim Cook says RAM expenses are ‘unsustainable’ and Apple is going to raise prices
           Source: The Verge
  [BULL +0.70] Is Apple Inc. (AAPL) A Good Stock To Buy Now?
           Source: Yahoo Entertainment
  [BULL +0.67] Prime Day Means Apple Deals on iPad, iPhone Cases, MagSafe Accessories, and More
           Source: Wired
  [BEAR -0.40] Apple's Cautious AI Strategy Could Have Been Its Smartest Move
           Source: CNET
  [BULL +0.38] iPhone 18 Pro's Camera Upgrade Will Cost Apple 50% More
           Source: MacRumors

  Running price path simulation (3 years)...
  Sentiment adjustment: +0.00005 daily drift
  Daily avg return:     0.075%
  Daily volatility:     1.655%
  Annualised vol:       26.3%

  Year 1 forecast:
    Median price:       $329.97  (+16.3%)
    Bear case (10th):   $235.50  (-17.0%)
    Bull case (90th):   $457.72  (+61.3%)
    Prob. above today:  72.4%

  Year 2 forecast:
    Median price:       $385.33  (+35.8%)
    Bear case (10th):   $233.92  (-17.6%)
    Bull case (90th):   $625.41  (+120.4%)
    Prob. above today:  77.7%

  Year 3 forecast:
    Median price:       $439.34  (+54.8%)
    Bear case (10th):   $242.50  (-14.5%)
    Bull case (90th):   $798.79  (+181.5%)
    Prob. above today:  82.5%


## Installation

```bash
git clone https://github.com/jayantrathi/valuation-tool
cd valuation-tool
pip install -r requirements.txt
python -m textblob.download_corpora
```

Create a `.env` file in the project folder:



Get a free API key at [newsapi.org](https://newsapi.org/register)

## Usage

```bash
python valuation_v3.py
```

Enter any stock ticker when prompted — AAPL, TSLA, KO, NVDA, MSFT, etc.

## How the blended price target works

| Method | Weight |
|---|---|
| DCF | 35% |
| Monte Carlo | 35% |
| Multiples | 30% |

## Limitations and future improvements

- DCF methodology undervalues hypergrowth companies (e.g. NVDA) whose 
  growth rates exceed the model's assumptions
- TextBlob sentiment analysis reads word-level polarity rather than 
  financial context — a future upgrade would use FinBERT, a sentiment 
  model trained specifically on financial news
- Price path simulation uses historical volatility which assumes future 
  volatility matches the past — this breaks down around major market events
- Model does not account for macroeconomic factors, interest rate changes, 
  or geopolitical risk

## Built with

- [yfinance](https://github.com/ranaroussi/yfinance) — live financial data
- [NewsAPI](https://newsapi.org) — news headlines
- [TextBlob](https://textblob.readthedocs.io) — sentiment analysis
- [NumPy](https://numpy.org) / [Pandas](https://pandas.pydata.org) — data processing
- [Matplotlib](https://matplotlib.org) — visualisation
