# GradePredict AI project guidance

This is an ARTI303 university final project. Students must understand, explain,
and demonstrate every important part themselves.

- Prefer simple Python, descriptive names, focused functions and logical modules.
- Avoid unnecessary abstractions, clever expressions and duplicated logic.
- Add short English comments for important decisions, especially cleaning and leakage prevention.
- Preserve the full workflow, validation, configuration, logging, documentation and tests.
- Explain why each change helps the project or makes it easier to present.
- When Phase 2 is authorized, begin with beginner-friendly, interpretable methods.
- Never modify the original dataset or the existing virtual environments.
- Windows Application Control blocks scientific components: do not execute blocked
  libraries or disable/bypass security. Keep scientific tests opt-in until resolved.
- Use the working `.venv_test` with `-B`. Do not install or alter environments.
- Active data is data/student-por.csv (semicolon CSV);
  features studytime/G1/G2 predict G3 after period two on a 0-20 grade scale.
- Keep zero final grades, original data, backup archives and comparison evidence.
- Optional Random Forest requires explicit opt-in; its tree component is blocked.
- Do not start a new phase or push to GitHub unless the user authorizes it.
