# Phase 8 Implementation Spec

## Goal

Run a parallel strategy-incubator pipeline for generating and filtering new candidate strategies.

## Depends On

- Phase 7

## Owns

- Tournament model (built on `DEAP` evolutionary primitives)
- Incubation phases
- Incubator fitness
- Mutation and crossover generators (reference: `sklearn-genetic-opt` callback architecture, adaptive scheduling)
- Regime-conditioned search
- Fossil record
- Population dynamics

## Contracts

- Incubator candidates are isolated from production promotion until they pass the shared promotion process
- Generation methods plug into a common tournament contract

## Non-Goals

- Direct production deployment of incubator outputs

## Acceptance Criteria

- Candidate strategies can be generated, evaluated, ranked, and archived
- Population diversity and graduation metrics are observable
- Incubator output feeds the broader evolution system without bypassing validation

