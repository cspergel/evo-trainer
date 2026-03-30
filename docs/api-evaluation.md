# Congressional & Political Trading Data API Evaluation

**Date:** 2026-03-29
**Purpose:** Evaluate data providers for congressional/political trading signals to feed into Evolve-Trader AI's evolutionary strategy engine.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Data Completeness | High | Fields: trade_date, filing_date, ticker, member name, party, state, committee membership |
| Historical Depth | Medium | How far back does data extend? |
| Update Latency | High | Time between STOCK Act filing and data availability |
| API Quality | High | REST design, rate limits, auth method, response format |
| Reliability | High | Uptime, known issues, maintenance track record |
| Cost | Medium | Free tier viability, paid tier value for prototyping vs. production |

---

## Congressional Trading Data Providers

### 1. Capitol Trades

| Attribute | Details |
|-----------|---------|
| **Website** | [capitoltrades.com](https://www.capitoltrades.com/) |
| **Data Completeness** | Excellent. Covers trade_date, filing_date, ticker, member name, party, state, asset type, trade size range. Committee data cross-referenced via ProPublica Congress API. |
| **Historical Depth** | Data from ~2019 onward (STOCK Act disclosures). |
| **Update Latency** | Typically within 1-2 business days of eFD/Clerk publication. |
| **API Quality** | **No public API.** Capitol Trades is a web platform only. Data access requires scraping via third-party tools (Apify actors, ScrapingBee, or custom scrapers). There is also an open-source MCP server on GitHub for extraction. |
| **Reliability** | Web UI is stable, but scraping is inherently fragile -- layout changes break scrapers. |
| **Cost** | Free to browse. Scraping via Apify: ~$5/month platform credit (free tier) for moderate volume. ScrapingBee charges per API call. |
| **Verdict** | Rich data but no official API makes it unsuitable as a primary programmatic source. Useful as a manual validation/cross-reference tool. |

---

### 2. Quiver Quantitative

| Attribute | Details |
|-----------|---------|
| **Website** | [quiverquant.com](https://www.quiverquant.com/) / [api.quiverquant.com](https://api.quiverquant.com/) |
| **Data Completeness** | Good. Fields include: ticker, member name (Representative), party, trade_date, transaction_date, trade type (Purchase/Sale), amount range. Senate and House tracked separately via `senate_trading()` and `house_trading()` methods, plus a combined `congress_trading()`. Does NOT include committee membership natively. |
| **Historical Depth** | January 2016 to present. Covers ~1,800 US equities. |
| **Update Latency** | Daily frequency. Scraped from SEC/eFD reports; typically available within 1-3 days of public filing. |
| **API Quality** | REST API with JSON responses. Bearer token authentication. Well-documented at `api.quiverquant.com/docs/`. OpenAPI spec available. |
| **Rate Limits** | Not publicly documented per-tier, but reasonable for programmatic use. |
| **Reliability** | Established provider (since ~2020). Used by QuantConnect as an official data source. Active GitHub presence. |
| **Cost** | **Hobbyist:** $10/month (Tier 1 data -- includes congress trading). **Trader:** $75/month (Tier 1 + Tier 2). **Institution:** Custom pricing. **Premium (web platform):** $25/month or $300/year. |
| **Python Package** | `quiverquant` on PyPI (v0.2.2). Install: `pip install quiverquant`. Official Quiver-Quantitative GitHub org. |
| **Verdict** | Best overall balance of data quality, API design, Python support, and price. Strong primary candidate. |

#### Python Package Details

```python
from quiverquant import quiver

q = quiver(token="YOUR_API_TOKEN")

# All congress trades (combined House + Senate)
df = q.congress_trading()

# Filter by ticker
df = q.congress_trading(ticker="NVDA")

# Senate only
df = q.senate_trading()

# House only
df = q.house_trading()

# 13F institutional data (contact for access)
df = q.sec13F()
```

---

### 3. AInvest API

| Attribute | Details |
|-----------|---------|
| **Website** | [docs.ainvest.com](https://docs.ainvest.com/reference/ownership/congress) |
| **Data Completeness** | Good. Response fields: `name`, `party`, `state`, `trade_date`, `filing_date`, `reporting_gap`, `trade_type`, `size`. No committee membership. |
| **Historical Depth** | From 2012 (STOCK Act inception) onward. |
| **Update Latency** | Not publicly documented. Appears to track eFD filings with moderate latency. |
| **API Quality** | REST API. Endpoint: `https://openapi.ainvest.com/open/ownership/congress`. Bearer token auth. JSON response wrapped in standard envelope (`status_code`, `status_msg`, `data`). |
| **Rate Limits** | Not publicly documented. |
| **Reliability** | Newer provider. Less community validation than Quiver or Finnhub. Documentation is sparse. |
| **Cost** | Pricing not publicly listed. Appears to have a free tier but limits are unclear. Contact required for details. |
| **Verdict** | Interesting data fields (includes `reporting_gap` which is useful for latency analysis), but opaque pricing and limited documentation make it a risk for a primary source. Worth monitoring. |

---

### 4. Finnhub

| Attribute | Details |
|-----------|---------|
| **Website** | [finnhub.io](https://finnhub.io/docs/api/congressional-trading) |
| **Data Completeness** | Moderate. Provides congressional trading data via dedicated endpoint. Fields include member name, ticker, transaction type, amount, dates. Party/state inclusion unclear from docs -- may require enrichment. No committee data. |
| **Historical Depth** | STOCK Act data from 2012+. |
| **Update Latency** | Not specifically documented for congressional data. General data updates are near-real-time for market data. |
| **API Quality** | Excellent REST API. API key via query parameter or header. JSON responses. Extensive documentation. Official Python client: `finnhub-python`. |
| **Rate Limits** | **Free:** 60 calls/minute. **Paid tiers:** Higher limits ($11.99-$99.99/month). |
| **Reliability** | Very well-established provider. Large user base. Widely used in fintech. |
| **Cost** | **Free:** 60 calls/min, basic data. **All-in-one:** $11.99/month. **Premium+:** Up to $99.99/month. Congressional trading may require a paid tier. |
| **Verdict** | Excellent general-purpose financial API with congressional data as one of many features. Good fallback if Quiver is unavailable. Generous free tier for prototyping. |

---

### 5. Financial Modeling Prep (FMP)

| Attribute | Details |
|-----------|---------|
| **Website** | [financialmodelingprep.com](https://site.financialmodelingprep.com/developer/docs/senate-trading-api) |
| **Data Completeness** | Good. Separate endpoints for Senate and House trades. Fields: date, asset/ticker, amount, price per share, transaction type, member name. Party and state available. Latest disclosures endpoint for real-time tracking. No committee data. |
| **Historical Depth** | STOCK Act filings from 2012+. |
| **Update Latency** | "Latest" disclosure endpoints suggest near-real-time ingestion of new eFD filings. |
| **API Quality** | REST API. API key via query parameter. JSON responses. Good documentation with endpoint examples. Separate Senate and House endpoints plus "latest" variants. |
| **Rate Limits** | **Free:** 250 calls/day. **Starter:** 300 calls/min. **Premium:** 750 calls/min. **Ultimate:** 3,000 calls/min. |
| **Bandwidth** | Free: 500MB/30d. Starter: 20GB. Premium: 50GB. Ultimate: 150GB. |
| **Reliability** | Well-established, widely-used financial data API. |
| **Cost** | **Free:** 250 calls/day (sandbox). **Starter:** ~$29/month (US data, 5-year history). **Premium:** ~$49/month (30-year history, UK/Canada). **Ultimate:** ~$99/month (global, transcripts, 13F). Congressional trading available on Starter+. |
| **Verdict** | Strong contender. Good data, reasonable pricing, well-documented API. Starter plan is affordable for prototyping. Separate Senate/House endpoints are a nice design choice. |

---

### 6. Unusual Whales

| Attribute | Details |
|-----------|---------|
| **Website** | [unusualwhales.com](https://unusualwhales.com/developers) / [api.unusualwhales.com/docs](https://api.unusualwhales.com/docs) |
| **Data Completeness** | Excellent. Congressional trading is a flagship feature. Includes member name, party, state, ticker, trade details, filing dates. Also provides aggregate analysis, congress trading reports, and "congress performance" metrics. |
| **Historical Depth** | STOCK Act data; publishes annual Congress Trading Reports (2023, 2024, 2025 editions). |
| **Update Latency** | Near-real-time alerts for new congressional filings. Push notifications available. |
| **API Quality** | REST + WebSocket + Kafka + MCP. Bearer token auth (header only, no query params). 100+ endpoints. OpenAPI spec available (YAML). JSON responses. Very modern API design. |
| **Rate Limits** | Not publicly documented per tier. |
| **Reliability** | Well-known platform. Active development. Large user community. |
| **Cost** | **Expensive.** Basic plan: ~$50/month. Historical options data: $250/month. API access appears to require premium subscription. Enterprise pricing available. Prices increased in May 2025. |
| **Verdict** | Best-in-class congressional data with excellent analysis features, but significantly more expensive than alternatives. Overkill for a prototype; potentially valuable at production scale if options flow and dark pool data are also needed. |

---

## Comparison Matrix: Congressional Trading Providers

| Provider | Data Quality | Historical | Latency | API Design | Free Tier | Cost (Paid) | Python Pkg |
|----------|-------------|------------|---------|------------|-----------|-------------|------------|
| Capitol Trades | A | B+ | B+ | F (none) | Browse only | Scraping cost | No |
| **Quiver Quant** | **A-** | **A (2016+)** | **B+** | **A** | **No** | **$10/mo** | **Yes** |
| AInvest | B+ | A (2012+) | B | B+ | Unclear | Unclear | No |
| Finnhub | B+ | A (2012+) | B | A+ | 60 req/min | $12-100/mo | Yes |
| **FMP** | **A-** | **A (2012+)** | **A-** | **A** | **250/day** | **~$29/mo** | **Yes** |
| Unusual Whales | A+ | A | A | A+ | No | $50+/mo | No |

---

## Institutional / 13F Data Sources

### WhaleWisdom

| Attribute | Details |
|-----------|---------|
| **Website** | [whalewisdom.com](https://whalewisdom.com/) |
| **Data** | 13F filings, institutional ownership, hedge fund portfolio tracking. |
| **API** | API access available for automated 13F queries. Last 9 quarters free; older data requires subscription. |
| **Historical Depth** | Extensive 13F history. |
| **Cost** | Free (limited to 9 quarters). Paid API pricing requires contacting sales. Third-party scraping via Apify: ~$1.50/1,000 results. |
| **Verdict** | Gold standard for 13F data browsing. API access exists but pricing is opaque. Best as a research tool rather than a programmatic feed unless budget allows. |

### HedgeFollow

| Attribute | Details |
|-----------|---------|
| **Website** | [hedgefollow.com](https://hedgefollow.com/) |
| **Data** | 13F institutional ownership, hedge fund holdings by stock. |
| **API** | **No public API.** Web platform only. Would require scraping for programmatic access. |
| **Historical Depth** | Covers historical 13F filings. |
| **Cost** | Free to browse. |
| **Verdict** | Useful for manual research. Not viable as a data feed without scraping. |

### TIKR

| Attribute | Details |
|-----------|---------|
| **Website** | [tikr.com](https://www.tikr.com/) |
| **Data** | Institutional 13F holdings, financials, global stock data, fundamental analysis. Tracks institutional disclosures globally (US, Europe, Asia, Australia). |
| **API** | **No API available.** Explicitly stated: "TIKR does not currently offer an API or Excel plug-in, given agreements with their various data providers." |
| **Historical Depth** | Extensive, global coverage. |
| **Cost** | Free tier with limited access. Paid plans available (see tikr.com/pricing). |
| **Verdict** | Excellent web research platform. Completely unsuitable for programmatic integration due to no API. |

### Fintel

| Attribute | Details |
|-----------|---------|
| **Website** | [fintel.io](https://fintel.io/) / [developers.fintel.io](https://developers.fintel.io/) |
| **Data** | 13F/NPORT institutional ownership, fund tracking, insider trading, short interest, beneficial ownership (13D/G). Complete ownership timelines. |
| **API** | Developer API available at `developers.fintel.io`. RESTful endpoints for fund tracking, institutional ownership, smart money signals. |
| **Historical Depth** | Complete 13F ownership history. |
| **Cost** | Pricing not publicly listed. Contact required for API access pricing. Web platform has free and premium tiers. |
| **Verdict** | Comprehensive institutional data with a developer API. Good option if budget allows. Pricing opacity is a drawback for evaluation. |

### 13F Data Comparison

| Provider | Has API | 13F Depth | Cost | Programmatic Use |
|----------|---------|-----------|------|-----------------|
| WhaleWisdom | Yes | Extensive | Opaque | Possible |
| HedgeFollow | No | Good | Free (web) | Scraping only |
| TIKR | No | Extensive | Paid | Not possible |
| Fintel | Yes | Extensive | Opaque | Possible |
| FMP (13F endpoint) | Yes | Good | ~$99/mo (Ultimate) | Yes |
| Finnhub (13F endpoint) | Yes | Good | Paid tier | Yes |

**For 13F/institutional data:** FMP (Ultimate tier) or Finnhub paid tiers are the most practical choices since they bundle 13F data with congressional trading in a single API and have transparent pricing.

---

## Python Package Investigation: Quiver Quantitative

### Package: `quiverquant` (PyPI)

- **Package name:** `quiverquant` (NOT `quiver-quantitative` or `quiver`)
- **Version:** 0.2.2
- **Install:** `pip install quiverquant`
- **PyPI:** [pypi.org/project/quiverquant](https://pypi.org/project/quiverquant/)
- **GitHub:** [Quiver-Quantitative/python-api](https://github.com/Quiver-Quantitative/python-api)
- **Dependencies:** `pandas`, `requests`

### Confirmed Methods

| Method | Description |
|--------|-------------|
| `congress_trading(ticker, politician, recent)` | Combined House + Senate trades |
| `senate_trading(ticker)` | Senate trades only |
| `house_trading(ticker)` | House trades only |
| `sec13F(ticker, date, owner, period)` | Institutional 13F data (contact for access) |
| `sec13FChanges(ticker, date, owner, period)` | 13F position changes |
| `insiders(ticker)` | Insider trading data |
| `lobbying(ticker)` | Lobbying disclosure data |
| `gov_contracts(ticker)` | Government contract awards |

Note: A separate package called `quiver` (v0.1) exists on PyPI but is **unrelated** -- it is not the Quiver Quantitative package.

---

## Primary Data Sources (All Free)

All data for this system comes from public sources. No paid APIs required.

### Congressional Trading Data — Build Our Own Scrapers

Congressional trades are public record under the STOCK Act. ~200 politicians, ~few hundred filings/month. Small scraping scope.

| Source | Format | What We Get |
|--------|--------|-------------|
| **House Clerk** (disclosures-clerk.house.gov) | HTML/PDF | ~435 House members, periodic transaction reports. **Reference impl exists:** `congressional-trading` on PyPI ([github.com/ivanma9/CongressionalTrading](https://github.com/ivanma9/CongressionalTrading)) — Python 3.12+, FastAPI, scrapes ZIP indexes → XML → PDF via pdftotext. Last commit March 2026. Has rate limiting, circuit breaker, retry logic. |
| **Senate eFD** (efdsearch.senate.gov) | HTML/PDF | ~100 senators, financial disclosures (terms checkbox needs Selenium) |
| **Capitol Trades** (capitoltrades.com/trades) | HTML scraping | Pre-normalized: 34K+ trades, 199 politicians, reporting delay, party, state, beneficial owner. 3-year history. Best scraping target — they've already done the PDF parsing. |

**Build plan:** Use `congressional-trading` package or its scraping patterns as a starting point for House data. Scrape Capitol Trades as primary for pre-normalized data. Build Senate eFD scraper for Senate coverage. Daily polling is sufficient — filings are delayed 30-45 days from trade date anyway.

### SEC EDGAR Data — Free API (No Auth)

| Source | Format | What We Get | Rate Limit |
|--------|--------|-------------|------------|
| **EDGAR FULL-TEXT** (efts.sec.gov) | XML/SGML | 13F quarterly holdings, Form 4 insider trades | 10 req/sec, no auth |
| **EDGAR filing stream** | RSS/JSON | Real-time new filing notifications | Same |

Already planned for Phase 2. Machine-readable XML, no scraping needed.

### Market Price Data — Free Libraries

| Source | Format | What We Get | Limits |
|--------|--------|-------------|--------|
| **yfinance** (Python package) | pandas DataFrames | Historical + real-time stock prices, dividends, splits, fundamentals | No API key, no hard rate limit (Yahoo may throttle) |
| **Alpha Vantage** (free tier) | JSON REST | 1-min intraday bars, daily OHLCV. Python SDK: `alpha_vantage` on PyPI ([github.com/RomelTorres/alpha_vantage](https://github.com/RomelTorres/alpha_vantage), 4.8K stars, community-maintained). | 25 req/day (backup only) |
| **Hyperliquid** (public API) | JSON REST | Crypto prices, orderbook | No auth, no hard limit |
| **FRED** (Federal Reserve) | JSON REST | Macro/economic indicators (rates, CPI, employment) | Free API key, 120 req/min |

**Build plan:** Use yfinance as primary price source (unlimited, battle-tested). Alpha Vantage as fallback for intraday data. Hyperliquid for crypto. FRED for macro signals.

### Committee Membership — Free APIs

| Source | Format | What We Get |
|--------|--------|-------------|
| **ProPublica Congress API** | JSON REST | Member → committee assignments, free, well-maintained |
| **unitedstates/congress** (GitHub) | Bulk JSON | Historical committee membership data |

Join on member name + date range to determine committee membership at time of trade. Critical for detecting trades correlated with committee-specific legislation.

## Recommendation

### Strategy: Build Everything From Public Sources ($0/month)

```
Phase 2:  Capitol Trades scraper (congressional trades)
          + EDGAR parsers (13F + Form 4)
          + yfinance (stock prices)
          + ProPublica Congress API (committee enrichment)

Phase 9:  Hyperliquid (crypto prices)
          + FRED (macro indicators)
          + Additional signal source scrapers as needed

If needed: Unusual Whales API (has free tier + MCP server) as
           cross-validation or if scraper maintenance becomes burdensome
```

### Paid APIs as Optional Cross-Validation

These remain available if we want to validate our scraped data or need faster integration:
- **Finnhub** — official Python SDK (`finnhub-python`, Apache 2.0, [github.com/Finnhub-Stock-API/finnhub-python](https://github.com/Finnhub-Stock-API/finnhub-python)). Only single SDK covering congressional trading + insider transactions + 13F + stock prices + crypto + fundamentals in one client. Free tier: 60 req/min for basic data (quotes, fundamentals). Congressional/insider data is premium (~$12+/mo). Best "upgrade path" if we later want one SDK for everything.
- **Unusual Whales** — REST API + MCP server, congress endpoints, free tier available
- **Quiver Quantitative** ($30/mo) — Python package with `congress_trading()`
- **Financial Modeling Prep** — free tier (250 calls/day) for price/fundamental cross-checks. Python SDK exists (`fmpsdk` on PyPI) but unmaintained since 2023. Congressional data requires expensive Ultimate plan. Not recommended.
