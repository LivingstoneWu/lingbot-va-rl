# Development Logging Workflow

## Purpose

Keep `CHANGELOG.md` as a concise, implementation-facing record that lets later sessions identify changed interfaces, new components, compatibility decisions, and completed verification.

## Workflow

1. Read the latest changelog section before modifying code.
2. During implementation, track public methods, classes, configuration keys, schemas, and behavioral contracts that change.
3. Before finishing, add one dated section for the development phase or cohesive change.
4. Record each new class/method and each modified method as a one-line entry.
5. Record verification commands and outcomes.
6. Keep planned work out of the changelog; log only changes present in the working tree.
7. Update an existing uncommitted entry instead of creating duplicate entries for follow-up fixes in the same phase.

## Entry Format

```text
## YYYY-MM-DD - Short Phase Name

### Added
- `ClassOrMethod`: one-line responsibility.

### Changed
- `Existing.method`: one-line behavioral or interface change.

### Removed
- `SymbolOrFile`: one-line reason or replacement.

### Verification
- `command`: result.
```

Omit empty headings.

## Detail Rules

- Name exact files, classes, methods, configuration keys, and checkpoint/schema versions.
- Describe observable behavior and ownership boundaries, not line-by-line implementation.
- Keep entries to one sentence whenever possible.
- Include pseudocode only for important multi-stage logic that is difficult to communicate accurately in one line.
- Do not paste large code snippets, logs, stack traces, generated files, or speculative design.
- Mention intentional exclusions when they prevent scope confusion.
- Never overwrite or reframe unrelated user-authored changelog entries.

## Completion Check

Before reporting completion:

```text
review diff
  -> update CHANGELOG.md
  -> verify entries match actual files
  -> run focused tests/checks
  -> record exact verification result
```
