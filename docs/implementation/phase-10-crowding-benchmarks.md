# Phase 10 Implementation Spec

## Goal

Detect source crowding, activate contrarian defenses, and validate the system under broad historical conditions.

## Depends On

- Phase 8
- Phase 9

## Owns

- Convergence scoring
- Source-independence scoring
- Contrarian skill family
- Historical crowding calibration
- Quiet-market validation
- Graceful-degradation validation
- Exploratory scenario packs
- Benchmark reporting

## Contracts

- Distributional validation is required
- Named historical scenarios are exploratory fixtures and regressions, not the primary acceptance gate

## Non-Goals

- Story-driven overfitting to famous trades

## Acceptance Criteria

- Crowding signals reduce exposure or activate hedges under high convergence
- Quiet markets do not trigger excessive churn
- The system degrades safely under signal failure
- Reports separate required failures from exploratory-scenario observations

