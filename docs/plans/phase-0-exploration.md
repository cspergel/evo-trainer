# Phase 0: Exploration & Foundation Setup — Detailed Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clone, run, and deeply understand both OpenSpace and AI-Trader codebases. Set up the Evolve-Trader repository with dev environment, CI, testing framework, Docker, and project structure. Document verified integration points. Evaluate API sources. Produce a cloud cost estimate.

**Architecture:** No new architecture in this phase — this is pure exploration and setup. We're building the foundation that every subsequent phase depends on.

**Tech Stack:** Python 3.11+, pyproject.toml, ruff, black, mypy, pytest, GitHub Actions, Docker, Docker Compose

**Parent plan:** [`2026-03-29-evolve-trader-master-plan.md`](2026-03-29-evolve-trader-master-plan.md)

---

## Task 1: Clone and Explore OpenSpace

**Files:**
- Clone: `lib/openspace/` (git submodule or local clone)
- Create: `docs/integration-analysis.md` (started here, completed in Task 5)

**Step 1: Clone the OpenSpace repository**

```bash
cd lib/
git clone https://github.com/HKUDS/OpenSpace.git openspace
```

**Step 2: Read the README and identify setup instructions**

Read `lib/openspace/README.md` and any setup/install docs. Document:
- Python version requirements
- Dependencies (requirements.txt or pyproject.toml)
- Environment variables needed
- LLM provider configuration (LiteLLM setup)

**Step 3: Install dependencies and run the test suite**

```bash
cd lib/openspace
pip install -e .  # or follow their install instructions
pytest  # or their test command
```

Expected: All tests pass. If not, document failures and workaround.

**Step 4: Explore the SKILL.md format**

Read actual SKILL.md files in the OpenSpace codebase. Document:
- What fields exist in a SKILL.md?
- How is the file parsed? (find the parser code)
- Can we add custom fields (entry logic, exit logic, target regime, performance characteristics)?
- If not extensible, what's the modification path?

Record findings in `docs/integration-analysis.md` under "## OpenSpace SKILL.md Format".

**Step 5: Trace the FIX/DERIVED/CAPTURED evolution flow**

Read the source code to understand:
- How does OpenSpace decide when to trigger FIX? (what metric/threshold?)
- How does DERIVED work? (what triggers specialization?)
- How does CAPTURED work? (how is emergent behavior detected?)
- Where is the post-execution analyzer? What interface does it use?
- Is evaluation binary (success/failure) or can we plug in custom metrics?

Record findings under "## OpenSpace Evolution Mechanics".

**Step 6: Trace the Version DAG**

Find the code that tracks skill lineage:
- How are parent→child relationships stored?
- What metadata is recorded per evolution event?
- Can we query the DAG for analysis?
- What storage backend does it use?

Record findings under "## OpenSpace Version DAG".

**Step 7: Trace Quality Monitoring & Promotion Gates**

Find the code that gates skill promotion:
- What checks must a skill pass before being promoted?
- Are there anti-loop guards? How do they work?
- Can we add custom promotion criteria (Sharpe threshold, drawdown limit)?

Record findings under "## OpenSpace Quality Monitoring".

**Step 8: Commit notes**

```bash
git add docs/integration-analysis.md
git commit -m "docs: OpenSpace integration analysis from code exploration"
```

---

## Task 2: Clone and Explore AI-Trader

**Files:**
- Clone: `lib/ai-trader/` (git submodule or local clone)
- Modify: `docs/integration-analysis.md`

**Step 1: Clone the AI-Trader repository**

```bash
cd lib/
git clone https://github.com/HKUDS/AI-Trader.git ai-trader
```

**Step 2: Read the README and identify setup instructions**

Document:
- Python version requirements
- Dependencies
- Environment variables (especially LLM API keys)
- Market data sources and any API keys needed

**Step 3: Install dependencies and run the test suite**

```bash
cd lib/ai-trader
pip install -e .  # or follow their install instructions
pytest  # or their test command
```

Expected: All tests pass. Document any failures.

**Step 4: Explore the REST API**

Read the source code to understand:
- What REST API endpoints does AI-Trader expose? (trade execution, market data, portfolio query, etc.)
- How does an LLM agent call these endpoints?
- What's the API interface contract?
- Can we add custom endpoints?

Record findings under "## AI-Trader REST API".

**Step 5: Trace the market data and paper trading system**

Find the code that handles market data serving and paper trading:
- How is market data loaded and served?
- What data sources does it use? (real-time feeds, historical APIs)
- What time granularity is supported? (daily, hourly, minute)
- How does the paper trading simulator work? (order execution, fill simulation, slippage)
- Can we feed historical data through the paper trading interface for backtesting?

Record findings under "## AI-Trader Market Data & Paper Trading".

**Step 6: Trace the portfolio engine**

Find the code that tracks positions and P&L:
- How are positions opened/closed?
- How is P&L calculated (realized vs unrealized)?
- What portfolio metrics are available? (we need Sharpe, drawdown, win rate)
- Is there a portfolio snapshot API we can query during simulation?

Record findings under "## AI-Trader Portfolio Engine".

**Step 7: Explore the multi-model support**

Find the LLM integration:
- How does AI-Trader switch between Claude/GPT-4o/Qwen? (uses OpenRouter, not LiteLLM)
- Is it configurable at runtime?
- Can different agents use different models simultaneously?

Record findings under "## AI-Trader Multi-Model Support".

**Step 8: Explore the crypto module**

Find the crypto market support:
- What crypto assets are included?
- How does it differ from the equity module?
- Is the portfolio engine crypto-aware (fractional units, 24/7 trading)?

Record findings under "## AI-Trader Crypto Support".

**Step 9: Commit notes**

```bash
git add docs/integration-analysis.md
git commit -m "docs: AI-Trader integration analysis from code exploration"
```

---

## Task 3: Identify Integration Points and Decision Gates

**Files:**
- Modify: `docs/integration-analysis.md`

**Step 1: Map the integration points**

Based on Tasks 1-2, document the concrete integration plan:

```markdown
## Integration Plan

### What OpenSpace gives us:
- [list specific modules/classes/functions we will use]

### What AI-Trader gives us:
- [list specific modules/classes/functions we will use]

### What we need to build (adapters):
- [list each adapter with: what it connects, why it's needed, estimated complexity]

### What we need to replace entirely:
- [list anything that won't work for our use case and must be rewritten]
```

**Step 2: Evaluate Decision Gates**

For each of these, record the answer found during exploration:

```markdown
## Decision Gate Results

### SKILL.md extensibility
- Finding: [can/cannot add custom fields]
- Decision: [extend schema / fork and modify / write adapter]

### Post-execution analyzer
- Finding: [binary only / pluggable metrics / needs replacement]
- Decision: [use as-is / write adapter / replace entirely]

### Historical replay custom evaluation
- Finding: [supports custom metrics during replay / end-only / needs modification]
- Decision: [use as-is / modify / build our own replay on top of their data]

### Anti-look-ahead enforcement for signal data
- Finding: [handles custom signal data / only handles market data / needs extension]
- Decision: [use as-is / extend / build our own enforcement]
```

**Step 3: Write the integration architecture summary**

One paragraph describing how the three codebases (OpenSpace, AI-Trader, Evolve-Trader) will fit together, based on actual findings.

**Step 4: Commit**

```bash
git add docs/integration-analysis.md
git commit -m "docs: integration plan and decision gate results"
```

---

## Task 4: Set Up Evolve-Trader Repository

**Files:**
- Create: `pyproject.toml`
- Create: `src/evolve_trader/__init__.py`
- Create: `src/evolve_trader/core/__init__.py`
- Create: `src/evolve_trader/signals/__init__.py`
- Create: `src/evolve_trader/strategies/__init__.py`
- Create: `src/evolve_trader/regime/__init__.py`
- Create: `src/evolve_trader/sizing/__init__.py`
- Create: `src/evolve_trader/orchestrator/__init__.py`
- Create: `src/evolve_trader/incubator/__init__.py`
- Create: `src/evolve_trader/dashboard/__init__.py`
- Create: `src/evolve_trader/execution/__init__.py`
- Create: `src/evolve_trader/monitoring/__init__.py`
- Create: `src/evolve_trader/discovery/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `.github/workflows/ci.yml`
- Create: `ruff.toml`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "evolve-trader"
version = "0.1.0"
description = "Evolutionary Strategy Discovery via Self-Evolving Agent Skills"
requires-python = ">=3.11"
dependencies = [
    # Core
    "pydantic>=2.0",
    "litellm>=1.0",
    # Data
    "sqlalchemy>=2.0",
    "alembic>=1.0",
    # API
    "fastapi>=0.100",
    "uvicorn>=0.20",
    "httpx>=0.24",
    # Analysis
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.1",
    "black>=23.0",
    "mypy>=1.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.black]
line-length = 100
```

**Step 2: Create ruff.toml**

```toml
line-length = 100
target-version = "py311"

[lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "SIM"]
```

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.env
.venv/
*.db
*.sqlite3
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

**Step 4: Create the project skeleton**

Create all `__init__.py` files listed above. Each should contain only:

```python
"""Evolve-Trader: [module name] module."""
```

**Step 5: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src/ tests/
      - run: black --check src/ tests/
      - run: mypy src/
      - run: pytest --cov=src/evolve_trader -v
```

**Step 6: Create a minimal passing test**

```python
# tests/unit/test_smoke.py
"""Smoke test to verify the project is set up correctly."""


def test_import():
    """Verify the evolve_trader package can be imported."""
    import evolve_trader
    assert evolve_trader is not None
```

**Step 7: Run the full CI pipeline locally**

```bash
pip install -e ".[dev]"
ruff check src/ tests/
black --check src/ tests/
mypy src/
pytest --cov=src/evolve_trader -v
```

Expected: All pass.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: initialize Evolve-Trader project skeleton with CI"
```

---

## Task 5: Docker & Docker Compose Setup

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `docker-compose.override.yml`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["uvicorn", "evolve_trader.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Step 2: Create docker-compose.yml**

```yaml
version: "3.9"

services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: evolve_trader
      POSTGRES_USER: evolve_trader
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://evolve_trader:dev_password@db:5432/evolve_trader
      LITELLM_API_KEY: ${LITELLM_API_KEY:-}
      ALPACA_API_KEY: ${ALPACA_API_KEY:-}
      ALPACA_SECRET_KEY: ${ALPACA_SECRET_KEY:-}
      ALPACA_BASE_URL: ${ALPACA_BASE_URL:-https://paper-api.alpaca.markets}
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    depends_on:
      - db

volumes:
  pgdata:
```

**Step 3: Create docker-compose.override.yml for dev hot-reload**

```yaml
version: "3.9"

services:
  app:
    command: uvicorn evolve_trader.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
```

**Step 4: Test Docker Compose**

```bash
docker compose up -d db
docker compose run --rm app pytest --cov=src/evolve_trader -v
docker compose down
```

Expected: PostgreSQL starts, tests pass inside container.

**Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml docker-compose.override.yml
git commit -m "feat: add Docker and Docker Compose for local development"
```

---

## Task 6: Evaluate Congressional Data APIs

**Files:**
- Create: `docs/api-evaluation.md`

**Step 1: Test each API**

For each of: Capitol Trades, Quiver Quantitative, AInvest, Finnhub, Financial Modeling Prep, Unusual Whales — make real API calls (free tier) and evaluate:

```markdown
# Congressional Data API Evaluation

## Evaluation Criteria
- Data completeness: does it include trade_date, filing_date, ticker, member, party, state, committee?
- Historical depth: how far back does data go?
- Update latency: how quickly after a STOCK Act filing does the data appear?
- API quality: REST? Rate limits? Auth method? Response format?
- Reliability: does it consistently return data? Any downtime observed?
- Cost: free tier limits? Paid tier pricing?

## Capitol Trades
[findings]

## Quiver Quantitative
[findings]

## AInvest API
[findings]

## Finnhub
[findings]

## Financial Modeling Prep
[findings]

## Unusual Whales
[findings]

## Recommendation
Primary: [choice] — because [reason]
Fallback: [choice] — because [reason]
```

**Step 2: Test the Quiver Python package**

```bash
pip install quiver-quantitative  # or whatever the actual package name is
python -c "import quiver; print(dir(quiver))"
```

If the package doesn't exist or doesn't have `congress_trading()`, document that and note we'll use direct REST calls.

**Step 3: Commit**

```bash
git add docs/api-evaluation.md
git commit -m "docs: congressional data API evaluation and recommendation"
```

---

## Task 7: Cloud Cost Estimate

**Files:**
- Create: `docs/cloud-cost-estimate.md`

**Step 1: Estimate resource requirements per phase**

```markdown
# Cloud Cost Estimate

## Resource Requirements by Phase

### Phase 0-1 (local only)
- No cloud costs. Development runs locally via Docker Compose.

### Phase 2-6 (minimal cloud)
- PostgreSQL: [managed DB or self-hosted on VPS?]
- Compute: single Python process + dashboard
- Estimated: [$/month]

### Phase 7-8 (orchestrator + incubator)
- Additional compute for incubator candidates (5-10 concurrent LLM-powered agents)
- Estimated: [$/month]

### Phase 9+ (full signal ingestion)
- Multiple FastAPI microservices for signal sources
- Estimated: [$/month]

## Provider Comparison

### Option A: Simple VPS (e.g., Hetzner, DigitalOcean)
- [specs and pricing]
- Pros: simple, cheap, full control
- Cons: self-managed, single point of failure

### Option B: AWS ECS/Fargate
- [specs and pricing]
- Pros: managed containers, auto-scaling
- Cons: more expensive, AWS complexity

### Option C: GCP Cloud Run
- [specs and pricing]
- Pros: pay-per-use, simple container deployment
- Cons: cold starts, less control

## LLM API Cost Estimate
- Per-trade strategy execution: ~[X] tokens × [Y] trades/day = $[Z]/month
- Evolution cycles: ~[X] per cycle × [Y] cycles/week = $[Z]/month
- Orchestrator: ~[X] per run × [Y] runs/month = $[Z]/month
- Incubator (5-10 candidates): ~[X] per candidate/day = $[Z]/month
- **Total estimated LLM cost: $[range]/month**

## Recommendation
[choice] for Phase 2-6, reassess at Phase 7.
```

**Step 2: Commit**

```bash
git add docs/cloud-cost-estimate.md
git commit -m "docs: cloud provider and LLM cost estimates"
```

---

## Task 8: Final Phase 0 Verification

**Files:**
- Modify: `docs/integration-analysis.md` (final review)

**Step 1: Verify all Phase 0 deliverables**

Run through the checklist:
- [ ] Both codebases running locally with all tests passing
- [ ] Integration analysis document complete with decision gate results
- [ ] Evolve-Trader repo initialized with CI pipeline passing
- [ ] Project skeleton with directory structure for all modules
- [ ] Dockerfile + docker-compose.yml working
- [ ] API source evaluation document complete with recommendation
- [ ] Cloud provider cost estimate complete

**Step 2: Run the full CI pipeline one more time**

```bash
ruff check src/ tests/
black --check src/ tests/
mypy src/
pytest --cov=src/evolve_trader -v
docker compose run --rm app pytest -v
```

Expected: Everything passes both locally and in Docker.

**Step 3: Review integration analysis against master plan assumptions**

Read through the master plan's Phase 1 section. For each assumption:
- Does the actual code support this? (verified in Tasks 1-3)
- If not, is the decision gate response documented?
- Does the master plan need updating based on what we found?

If the master plan needs updates, make them now before starting Phase 1.

**Step 4: Final commit**

```bash
git add -A
git commit -m "docs: Phase 0 complete — all deliverables verified"
```

---

## Summary: Phase 0 Task Sequence

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Clone and explore OpenSpace | — |
| 2 | Clone and explore AI-Trader | — |
| 3 | Identify integration points and decision gates | 1, 2 |
| 4 | Set up Evolve-Trader repository | — |
| 5 | Docker & Docker Compose setup | 4 |
| 6 | Evaluate congressional data APIs | — |
| 7 | Cloud cost estimate | — |
| 8 | Final verification | 1-7 |

**Parallelizable:** Tasks 1, 2, 4, 6, and 7 are independent and can run in parallel. Tasks 3 and 5 have dependencies. Task 8 requires all others complete.
