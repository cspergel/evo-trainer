# Phase 2 Implementation Spec

## Goal

Move persistence to PostgreSQL and establish the first signal-ingestion framework.

## Depends On

- Phase 1

## Owns

- PostgreSQL schema and repositories
- `SignalEvent` type system
- Decay library
- Source registration framework
- EDGAR 13F ingestion (via `pibou-filings` package — parses 13F-HR XML into structured holdings with CUSIPs)
- EDGAR Form 4 ingestion (via `pibou-filings` package — parses Section 16 XML into transaction-level data)
- Congressional trading ingestion: House via `congressional-trading` PyPI package, Senate via Capitol Trades scraper, committee enrichment via ProPublica Congress API
- Basic regime classifier
- Domain-to-ORM converters (TradeResult↔TradeLog, SignalEvent↔SignalEventRecord, EvolutionEvent↔EvolutionEventRecord)

## Deferred (tracked for later phases)

- CUSIP-to-ticker mapping layer (OpenFIGI API) — add when 13F live polling is wired
- Real-time filing alerts (SEC EDGAR RSS) — add when live signal polling is needed
- Migration script from Phase 1 JSONL to PostgreSQL — add when PostgreSQL is in production use
- PostgreSQL-backed `LLMUsageLogger` swap — DB repository exists, logger backend swap deferred
- Senate data (Capitol Trades scraper) — noted in congressional source TODO
- Committee enrichment (ProPublica Congress API) — noted in congressional source TODO
- Signal source `fetch_signals()` live polling — parsers/converters exist, polling deferred to Phase 6+

## Contracts

- Phase 1 logger interface remains stable
- Signal ingestion writes typed records with source metadata and decay config
- Database access goes through typed repositories

## Non-Goals

- Advanced source weighting
- Discovery engine
- Live execution

## Acceptance Criteria

- Signal sources register and emit typed events
- Basic regime classifier consumes those events
- ORM models cover all domain types with typed repositories
- Converters bridge in-memory types to database records
