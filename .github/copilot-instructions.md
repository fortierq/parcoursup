# Project Guidelines

## Refactoring

- Prefer the simplest correct implementation that preserves behavior.
- When changing code, look for opportunities to simplify control flow, naming, and structure instead of layering on extra complexity.
- Remove dead code when it is safe to do so: unused functions, obsolete branches, stale helpers, redundant comments, and unused imports.
- Factor repeated logic into shared helpers when that reduces duplication without making the code harder to follow.
- Avoid duplication across code, tests, and data-processing logic; keep one clear source of truth when possible.
- Do not keep compatibility shims, indirection, or defensive branches unless there is a demonstrated need.
- Favor small, focused functions and explicit code over clever abstractions.
- If duplication or dead code cannot be removed safely within the current task, call it out clearly.