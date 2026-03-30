# Tooling Inventory — External Packages & Reference Architectures

All packages and reference repos evaluated during Phase 0. Organized by use and phase.

---

## Direct Dependencies (in pyproject.toml)

| Package | PyPI | Use | Phase |
|---------|------|-----|-------|
| `quantstats` | [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) (6.9K stars) | ~70 financial metrics (Sharpe, Sortino, drawdown, Monte Carlo, HTML tearsheets) | 1+ |
| `empyrical` | pypi.org/project/empyrical | Lightweight programmatic metrics (alpha, beta, annual return) | 1+ |
| `yfinance` | pypi.org/project/yfinance | Historical + real-time stock prices, dividends, splits. No API key. | 1+ |
| `pibou-filings` | [Pierre-Bouquet/pibou-filings](https://github.com/Pierre-Bouquet/pibou-filings) (97 stars, March 2026) | 13F-HR + Form 4 XML parsing with EDGAR rate-limiting | 2 |
| `congressional-trading` | [ivanma9/CongressionalTrading](https://github.com/ivanma9/CongressionalTrading) (March 2026) | House Clerk PDF scraping with rate limiting, circuit breaker | 2 |
| `ta` | pypi.org/project/ta | 150+ technical indicators (RSI, MACD, Bollinger, etc.) for seed strategies and signal features | 1+ |
| `PyPortfolioOpt` | pypi.org/project/PyPortfolioOpt | Mean-variance optimization, Black-Litterman, risk parity, efficient frontier | 4 |
| `alpaca-py` | pypi.org/project/alpaca-py | Official Alpaca SDK for paper + live trading | 6, 11 |
| `APScheduler` | pypi.org/project/APScheduler | Job scheduling for polling scrapers, signal refresh intervals | 2+ |
| `deap` | pypi.org/project/deap | Evolutionary algorithm primitives (tournament, crossover, mutation) | 8 |
| `pydantic` | pypi.org/project/pydantic | Data validation / schema | 1+ |
| `litellm` | pypi.org/project/litellm | Multi-model LLM routing (used by OpenSpace) | 1+ |
| `sqlalchemy` | pypi.org/project/sqlalchemy | Database ORM | 2+ |
| `fastapi` | pypi.org/project/fastapi | Dashboard API | 5+ |

## Reference Architectures (study patterns, reimplement — all MIT licensed)

### Replay Harness (Phase 1)

| Repo | What to borrow |
|------|----------------|
| [DanRedelien/futures-backtesting-engine](https://github.com/DanRedelien/futures-backtesting-engine) (March 2026, MIT) | No-lookahead 6-phase bar loop (Order Exec → Mark-to-Market → WFO Hook → Halt → Signal Gen → EOD Close). FastBar numpy optimization (70x speedup). Strategy registry via importlib. Walk-forward via Optuna with pruning. Three-tier risk halts. Redis/RQ job dispatch for parallel evaluation. |
| [zachisit/july-backtester](https://github.com/zachisit/july-backtester) (March 2026, MIT) | Walk-forward with overfitting detection (IS > 0 and OOS < 0 = overfitted). Monte Carlo 1,000-path robustness scoring. SQN/R-Multiple metrics. Decorator-based `@register_strategy` plugin system. Multiprocessing with immutable globals pattern. VIX regime heatmaps. |

### SEC Filing Alerts (Phase 2)

| Repo | What to borrow |
|------|----------------|
| [ryansmccoy/py-sec-edgar](https://github.com/ryansmccoy/py-sec-edgar) (March 2026) | RSS feed workflow for real-time new filing alerts. Full-index and daily-index crawlers for historical backfill. CLI filtering by ticker/form type/date range. |

### Incubator Evolution (Phase 8)

| Repo | What to borrow |
|------|----------------|
| [rodrigo-arenas/Sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt) (357 stars) | Callback architecture (early stopping, TensorBoard, MLflow, checkpointing). Adaptive mutation/crossover scheduling. `Continuous`/`Integer`/`Categorical` parameter space definitions. Warm start with known-good configs. Duplicate caching. |

## Paid API Upgrade Paths (optional, not needed for prototyping)

| Package | What it covers | Cost |
|---------|---------------|------|
| `finnhub-python` ([Finnhub-Stock-API/finnhub-python](https://github.com/Finnhub-Stock-API/finnhub-python), Apache 2.0) | Only single SDK covering congressional + insider + 13F + prices + crypto + fundamentals | Free tier: 60 req/min basic data. Premium ~$12+/mo for congressional/insider |
| `alpha_vantage` ([RomelTorres/alpha_vantage](https://github.com/RomelTorres/alpha_vantage), 4.8K stars) | Alpha Vantage wrapper for intraday data | Free: 25 req/day |
| Unusual Whales API | Congressional trading REST API + MCP server | Free tier available |
| `quiverquant` | Congressional trading pandas DataFrames | $30/mo |

## Evaluated and Rejected

| Repo | Why |
|------|-----|
| `marketcalls/vectorbt-backtesting-skills` | No license — can't reuse legally |
| `lukerosiak/pysec` | Dead since 2014, targets XBRL not 13F/Form 4 |
| `galibin24/SEC-EDGAR-python-scraper` | Student project, abandoned 2019, undocumented |
| `kaushalshetty/FeatureSelectionGA` | Too limited (binary-only), not maintained. Use DEAP directly. |
| `daxm/fmpsdk` | Unmaintained since 2023, doesn't cover congressional/insider endpoints |
| `financial-datasets/mcp-server` | Fundamentals-only (no congressional/insider/13F), requires paid API key |
| `timothycarambat/senate-stock-watcher-data` | Dead since March 2021 |
