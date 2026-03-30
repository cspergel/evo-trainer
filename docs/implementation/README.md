# Evolve-Trader Implementation Docs

This directory is the implementation-spec layer derived from `docs/plans`.

Use these docs for build execution. Use the planning docs for history, rationale, and exploratory detail.

## Structure

- `master-implementation-spec.md`: system-wide contracts, phase graph, and cross-cutting rules
- `phase-0-foundation.md` through `phase-12-extensions.md`: per-phase implementation specs

## Rules

- When a phase doc conflicts with an older planning snippet, follow the implementation doc.
- Shared contracts are defined once in the master spec and extended, not redefined, by phases.
- Acceptance criteria are the minimum exit bar for each phase.
- Exploratory scenarios are not production acceptance gates unless explicitly marked as required.

