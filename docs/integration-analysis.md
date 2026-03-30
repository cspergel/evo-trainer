# Integration Analysis — OpenSpace + AI-Trader

> Generated from Phase 0 code exploration. All findings verified against source code.
>
> **Pinned upstream revisions (git submodules):**
> - OpenSpace: `11bdf128d9b53a4107aec1d3098fa04ab808a700` (HKUDS/OpenSpace)
> - AI-Trader: `3b3169b756002518b752baae994a2d1bdbb70600` (HKUDS/AI-Trader)

---

## OpenSpace SKILL.md Format

### Structure
Every skill is a **directory** containing a `SKILL.md` file plus optional auxiliary files:
```
my-skill/
├── SKILL.md          # YAML frontmatter + markdown body
├── .skill_id         # Auto-created sidecar (persistent ID)
└── [auxiliary files]  # Optional scripts, configs, examples
```

### Schema
**Required frontmatter fields:** `name`, `description` only.
**Body:** Free-form markdown (agent instructions). No formal schema — agent interprets as natural language.

### Parser
- **File:** `openspace/skill_engine/skill_utils.py` lines 79-98
- Functions: `parse_frontmatter()`, `get_frontmatter_field()`, `set_frontmatter_field()`, `normalize_frontmatter()`
- Uses regex-based line-by-line parser (no PyYAML dependency)

### Extensibility
**Custom fields ARE supported.** The parser treats all key-value pairs equally — no schema validation. Custom fields are:
- Parsed into the dict successfully
- Persisted in database (`skill_records.lineage_content_snapshot`)
- Not enforced by the core system (informational only)

**System-managed fields** (added during evolution, not in source SKILL.md):
- `lineage_origin` (IMPORTED / FIXED / DERIVED / CAPTURED)
- `lineage_generation` (distance from root in version DAG)
- `lineage_change_summary` (evolution notes)

### Integration Impact
We can add trading-specific fields (`entry_logic`, `exit_logic`, `target_regime`, `expected_sharpe`, `risk_parameters`) directly to SKILL.md frontmatter. No fork needed. We will need to write our own validation layer that enforces these fields for trading skills.

---

## OpenSpace Evolution Mechanics

### Three Evolution Triggers

| Trigger | File | Timing | Initiator |
|---------|------|--------|-----------|
| Post-analysis | `evolver.py:257` | After task ends | ExecutionAnalyzer |
| Tool degradation | `evolver.py:290` | When tool quality drops | ToolQualityManager |
| Metric monitor | `evolver.py:414` | Periodic scan (every N executions) | Background task |

### Three Evolution Types (`types.py` lines 26-29)

```python
class EvolutionType(str, Enum):
    FIX      = "fix"       # Repair in-place (same name, same directory)
    DERIVED  = "derived"   # Enhanced version (new name, new directory)
    CAPTURED = "captured"  # Brand-new pattern (no parent)
```

### FIX Flow (`evolver.py:617-700`)
1. Load parent skill + failure context
2. LLM agent loop (max 5 iterations) generates patch
3. Patch applied (SEARCH/REPLACE, unified diff, or multi-file)
4. Validation: re-read SKILL.md, ensure valid YAML frontmatter
5. Persist: same `skill_id`, same directory, new version. Old version → `is_active=False`

### DERIVED Flow (`evolver.py:701-770`)
1. Load one or more parent skills (multi-parent composition supported)
2. Generate new skill name, create new directory
3. Apply LLM-generated changes
4. Persist: new `skill_id`, parents remain active, new skill is active
5. `generation = parent.generation + 1`

### CAPTURED Flow (`evolver.py:771-820`)
1. No parent skills — analyzing successful execution
2. Extract reusable pattern from execution recordings
3. Generate new skill from scratch
4. Persist: new `skill_id`, `parent_skill_ids=[]`, `generation=0`

### Post-Execution Analyzer Interface (`analyzer.py`)

**Input:**
```python
async def analyze_execution(
    task_id: str,
    recording_dir: str,
    execution_result: Dict[str, Any],
    available_tools: Optional[List[BaseTool]] = None,
) -> Optional[ExecutionAnalysis]
```

**Output:**
```python
@dataclass
class ExecutionAnalysis:
    task_id: str
    task_completed: bool
    execution_note: str
    tool_issues: List[str]
    skill_judgments: List[SkillJudgment]
    evolution_suggestions: List[EvolutionSuggestion]
    analyzed_by: str
```

**Evaluation is binary** (task_completed yes/no, skill_applied yes/no). Custom financial metrics must be injected into the analyzer prompt — not a pluggable metrics interface.

### Anti-Loop Guards
1. **Tool degradation:** State-driven tracking (`_addressed_degradations` dict). Cleared when tool recovers.
2. **Metric check:** Data-driven — skills need ≥5 selections before re-evaluation. Newly-evolved skills start at 0.
3. **Post-analysis:** One-shot per task. Evolution is optional (depends on LLM suggestion).
4. **Confirmation gate:** LLM is asked "Confirm this skill needs evolution?" before executing.

---

## OpenSpace Version DAG

### Storage Backend: SQLite
**Location:** `.openspace/openspace.db` (auto-created in project root)

### Key Tables (`store.py` lines 79-165)
- `skill_records` — Main skill identity + metadata
- `skill_lineage_parents` — Parent→child relationships (many-to-many)
- `execution_analyses` — Task execution analysis records
- `skill_judgments` — Per-skill assessment within an analysis
- `skill_tool_deps` — Tool dependencies
- `skill_tags` — Auxiliary tags

### Querying the DAG
```python
store.load_record(skill_id) → SkillRecord
store.get_versions(name) → List[SkillRecord]  # all versions, active + inactive
store.find_children(parent_skill_id) → List[str]
store.load_analyses(skill_id, limit=50) → List[ExecutionAnalysis]
store.find_skills_by_tool(tool_key) → List[str]
```

### Metadata Per Evolution Event
- `created_at`, `created_by` (model name)
- `origin` (FIXED/DERIVED/CAPTURED), `generation` (distance from root)
- `parent_skill_ids`, `change_summary`
- `content_snapshot` (full directory state), `content_diff` (unified diff)
- `source_task_id` (optional)
- Execution stats: `total_selections`, `total_applied`, `total_completions`, `total_fallbacks`

---

## OpenSpace Quality Monitoring

### Skill Metrics (`store.py`)
- `applied_rate` = selections where skill was used / total selections
- `completion_rate` = applied uses that led to task completion / total applied
- `effective_rate` = task completion / total selections
- `fallback_rate` = times not used + task failed / total selections

### Tool Metrics (`grounding/core/quality/types.py`)
- `success_rate`, `recent_success_rate` (rolling 100 executions)
- `consecutive_failures`, `penalty` (0.2-1.0 composite score)

### Promotion Gates
- No hard gates before DERIVED/CAPTURED creation — new skills start `is_active=True`
- FIX applies to already-active skills
- Rule-based screening thresholds: fallback_rate > 40%, completion_rate < 35%
- LLM confirmation required before executing evolution

### Custom Promotion Criteria
**Can be added** by modifying:
- `evolver.py` `_diagnose_skill_health` — add custom metric checks
- `evolver.py` `_validate_evolved_skill` — add post-evolution validation
- `quality/manager.py` — custom penalty functions

---

## AI-Trader MCP Toolchain

### CRITICAL FINDING: No MCP Tools
**AI-Trader does NOT expose MCP tools.** It provides a **REST API + WebSocket** interface.

### Agent Integration
- Agents register via `POST /api/claw/agents/selfRegister` → JWT token
- All operations via HTTP: trade execution, market data, portfolio query
- Notifications via HTTP polling (heartbeat) or optional WebSocket

### Core API Endpoints (`routes.py`)
1. **Trade Execution:** `POST /api/signals/realtime` — buy, sell, short, cover
2. **Market Data:** `GET /api/price` — current or historical price (1-min bars)
3. **Portfolio:** `GET /api/positions` — all positions + P&L + cash
4. **Signals:** `POST /api/signals/strategy`, `POST /api/signals/discussion`
5. **Notifications:** `POST /api/claw/agents/heartbeat`

### Skill Files
AI-Trader has its own `skills/` directory with 6 modules:
- `ai4trade` — authentication, signal feed
- `copytrade` — follow traders, copy positions
- `tradesync` — publish trading signals
- `heartbeat` — poll notifications
- `polymarket` — Polymarket public API
- `market-intel` — read-only market news/macro

---

## AI-Trader Historical Replay

### CRITICAL FINDING: Not a Backtest Engine
**AI-Trader is a paper trading simulator, NOT a historical replay system.**

- Market data is real-time only (Alpha Vantage for stocks, Hyperliquid for crypto)
- Historical prices can be queried with `executed_at` timestamp but there is no "replay a time period" mode
- No built-in walk-forward validation
- No corporate action handling (splits, dividends)

### Data Sources
| Market | Source | Auth | Granularity |
|--------|--------|------|-------------|
| US Stocks | Alpha Vantage | API key required | 1-minute bars |
| Crypto | Hyperliquid | Public (no auth) | 1-minute bars |
| Polymarket | Gamma + CLOB APIs | Public (no auth) | Current snapshot only |

### Anti-Look-Ahead
Implemented at execution level: historical prices queried with `executed_at` timestamp return data up to that minute. But this is per-trade, not a systematic replay harness.

---

## AI-Trader Portfolio Engine

### Position Schema (`database.py` lines 562-579)
- Fields: `agent_id`, `symbol`, `market`, `side` (long/short), `quantity` (REAL), `entry_price`, `current_price`, `opened_at`
- Markets: `us-stock`, `crypto`, `polymarket`
- Fractional units supported (REAL type)

### P&L Calculation
- **Unrealized:** Calculated on-demand from positions: `(current_price - entry_price) * quantity`
- **Realized:** Stored in `profit_history` table, recorded every 5 minutes via background task
- **Missing metrics:** No Sharpe, Sortino, max drawdown, or win rate. Would need custom calculation.

### Portfolio Snapshot
`GET /api/positions` returns current snapshot with all positions, P&L, and cash balance.

---

## AI-Trader Multi-Model Support

### FINDING: Uses OpenRouter, Not LiteLLM
- **File:** `market_intel.py` lines 20-23 — imports `openrouter` library
- Global model config via env vars: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- Supports any model on OpenRouter (Claude, GPT-4o, Qwen, etc.)
- Model selection is global, not per-agent

---

## AI-Trader Crypto Module (BITWISE10)

### FINDING: No BITWISE10 Module
- Tracks BTC ETF symbols: IBIT, FBTC, ARKB, BITB, HODL, BRRR, EZBC, BTCW
- Crypto supported via Hyperliquid (BTC, ETH, any Hyperliquid-listed token)
- Fractional units supported
- 24/7 trading supported (no market hours check for crypto)
- Portfolio engine is crypto-aware (separate `market` column)

---

## Test Suites

### OpenSpace: No Unit Tests
- No `pytest` directory or test files found
- Has benchmark evaluation suite (`gdpval_bench/`) with 220 real-world tasks
- Has showcase project (`showcase/`) demonstrating 60+ evolved skills

### AI-Trader: No Tests
- No test files, pytest config, or test infrastructure found

---

## Integration Plan

### What OpenSpace Gives Us
- Skill evolution engine (FIX/DERIVED/CAPTURED) — `skill_engine/evolver.py`
- Version DAG with full lineage tracking — `skill_engine/store.py`
- Skill discovery and registry — `skill_engine/registry.py`
- SKILL.md parser with extensible frontmatter — `skill_engine/skill_utils.py`
- Quality monitoring (skills + tools) — `grounding/core/quality/`
- Post-execution analyzer — `skill_engine/analyzer.py`
- LiteLLM integration for multi-model support — `pyproject.toml`

### What AI-Trader Gives Us
- Paper trading execution via REST API — `service/server/routes.py`
- Position tracking with multi-market support — `service/server/database.py`
- Market data from Alpha Vantage + Hyperliquid — `service/server/price_fetcher.py`
- Profit history recording — `service/server/tasks.py`
- Agent registration and authentication — JWT-based
- Market intel snapshots (LLM-generated) — `service/server/market_intel.py`

### What We Need to Build (Adapters)
1. **Historical replay harness** — AI-Trader doesn't do backtesting. We need to build a replay engine that feeds historical market data through our strategy evaluation pipeline. **High complexity.**
2. **REST-to-internal adapter** — Bridge between our internal `TradeIntent` contract and AI-Trader's REST API. Medium complexity.
3. **Financial metrics calculator** — Sharpe, Sortino, max drawdown, win rate. AI-Trader doesn't compute these. Medium complexity.
4. **Custom analyzer prompt injection** — Inject trading metrics into OpenSpace's post-execution analyzer so evolution decisions are finance-aware. Low complexity.
5. **StrategySkill validation layer** — Enforce trading-specific SKILL.md fields that OpenSpace doesn't validate. Low complexity.

### What We Need to Replace Entirely
1. **Walk-forward validation** — Neither upstream provides this. Must build from scratch for Phase 1.
2. **Portfolio-level risk enforcement** — AI-Trader has no immutable risk constraints. Our risk layer wraps around AI-Trader's execution API.
3. **Per-agent model selection** — AI-Trader's model config is global. We need per-strategy model routing (LiteLLM from OpenSpace side).

---

## Decision Gate Results

### SKILL.md Extensibility
- **Finding:** Can add custom fields freely. Parser accepts arbitrary key-value pairs.
- **Decision:** Extend schema with trading fields. Write our own validation layer. No fork needed.

### Post-Execution Analyzer
- **Finding:** Binary evaluation (task completed yes/no). Can inject custom context into the analyzer LLM prompt.
- **Decision:** Write adapter that injects financial metrics (Sharpe, drawdown, win rate) into the analyzer prompt. Keep OpenSpace's analyzer architecture, customize the prompt.

### Historical Replay / Custom Evaluation
- **Finding:** AI-Trader is NOT a backtest engine. It's a paper trading simulator with real-time data only.
- **Decision:** Build our own historical replay harness. Use AI-Trader's price fetcher as a data source adapter, but control the replay loop ourselves.

### Anti-Look-Ahead Enforcement for Signal Data
- **Finding:** AI-Trader enforces per-trade (historical price query returns data up to timestamp), but has no systematic replay with signal isolation.
- **Decision:** Build our own enforcement. Time-fence all signal data in our replay harness.

---

## Integration Architecture Summary

Evolve-Trader sits between OpenSpace and AI-Trader as an integration and extension layer. OpenSpace provides the skill evolution engine — its SKILL.md format, version DAG, and FIX/DERIVED/CAPTURED mechanics are used directly with trading-specific field extensions. AI-Trader provides paper trading execution and market data via its REST API, but it is NOT used as a backtest engine. The most significant finding is that we must build our own historical replay harness for walk-forward validation (Phase 1's core deliverable), since neither upstream provides backtesting capability. Financial metrics (Sharpe, drawdown, win rate) must be computed by Evolve-Trader and injected into OpenSpace's evolution pipeline so that strategy improvement decisions are finance-aware rather than binary task-completion based.
