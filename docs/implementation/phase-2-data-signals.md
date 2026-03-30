# Phase 2 Implementation Spec

## Goal

Move persistence to PostgreSQL and establish the first signal-ingestion framework.

## Depends On

- Phase 1

## Owns

- PostgreSQL schema and repositories
- Migration path from Phase 1 persistence
- `SignalEvent` type system
- Decay library
- Source registration framework
- EDGAR 13F ingestion (via `pibou-filings` package — parses 13F-HR XML into structured holdings with CUSIPs)
- EDGAR Form 4 ingestion (via `pibou-filings` package — parses Section 16 XML into transaction-level data)
- CUSIP-to-ticker mapping layer (OpenFIGI API, free)
- Real-time filing alerts (reference: `py-sec-edgar` RSS workflow)
- Congressional trading ingestion: House via `congressional-trading` PyPI package, Senate via Capitol Trades scraper, committee enrichment via ProPublica Congress API
- Basic regime classifier
- PostgreSQL-backed `LLMUsageLogger`

## Contracts

- Phase 1 logger interface remains stable
- Signal ingestion writes typed records with source metadata and decay config
- Database access goes through typed repositories

## Non-Goals

- Advanced source weighting
- Discovery engine
- Live execution

## Acceptance Criteria

- SQLite/file-backed Phase 1 data migrates without duplication
- Signal sources register and emit typed events
- Basic regime classifier consumes those events
- Logger costs and budget metrics are queryable from PostgreSQL

