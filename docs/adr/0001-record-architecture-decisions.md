# 1. Record architecture decisions

- Status: accepted
- Date: 2026-10-13

## Context
Retrieval design choices (which index, which fusion, what to measure) are easy to forget and hard to reverse once data has been indexed.

## Decision
Record significant decisions as short ADRs (Michael Nygard format) in `docs/adr/`.

## Consequences
Every non-obvious choice has a written rationale that reviewers can challenge. ADRs are append-only: a reversed decision gets a new ADR that supersedes the old one.
