# Python and ML rules

> **Status:** REFERENCE · **Updated:** 2026-09-07
> Coding and testing conventions for Python changes.

**Follow PEP 8 + ML reproducibility best practices.**

Environment, compatibility, and Git constraints remain in [engineering rules](engineering.md).

## Coding Standards

| Category            | Rule                                                                      |
| ------------------- | ------------------------------------------------------------------------- |
| **Type hints**      | Add type hints to functions where applicable                              |
| **Naming**          | Use `snake_case` for functions/vars and `PascalCase` for classes          |
| **Device handling** | Training and inference code must check CUDA availability and support CPU  |
| **Random seeds**    | Seed `torch`, `numpy`, and `random` in reproducible experiment entrypoints |
| **Logging**         | Use `logging`, not `print`, in library and long-running production code   |
| **Tests**           | Use `pytest` fixtures and meaningful value/shape/edge-case assertions     |
| **Configs**         | Put experiment hyperparameters in YAML under `config/`                    |
| **Error handling**  | Raise explicit exceptions; never fail silently                            |
| **Assertions**      | Use `assert` for dev/debug only; raise for production validation          |
| **Dict access**     | Use `dict[key]` for required fields; do not hide missing required values  |

## Formatting, imports, and docstrings

- Use four spaces per indentation level and a 100-character formatting line length.
  Respect [.gitattributes](../../.gitattributes) for line endings and file handling.
- Follow [pyproject.toml](../../pyproject.toml) for tool configuration instead of duplicating
  its rule lists. Ruff targets Python 3.11; package compatibility is declared in that file.
  Black and isort are also configured, with isort using the Black profile. Apply formatting
  only to task-related changes; do not run broad automatic fixes on unrelated code.
- Order imports as future imports, standard library, third-party packages, first-party
  `fragnnet`, then relative imports. Separate groups and sort by library name.
- Use `import torch.nn.functional as F` and `import lightning as L` when importing those
  modules. Use `F` and `L` consistently in their consumers.
- Use Google-style docstrings for public modules, classes, methods, and functions.
  Private helpers and obvious getters may omit docstrings. Do not document `self` or `cls`.
- Document tensor shapes in arguments and returns, optional defaults and their meaning,
  and mathematical operations. Include sections as needed in this order: summary,
  extended description, Args, Returns, Yields, Raises, Attributes, Example, Note, See Also.
- Prefix intentionally unused variables with `_`; follow the configured lint exceptions.

## Testing conventions

- Put tests in `tests/test_<module_or_feature>.py`, grouping related tests by module,
  feature, or class. Keep distinct major components in separate files.
- Use fixtures for reusable setup. For model, data, or utility features, include a known
  molecule/example and an edge case appropriate to the behavior.
- Test expected values and shapes, not only existence or nonempty length. Cover model
  input/output pairs, loss values (including zeros or large values), loader shapes and
  types, utility outputs, and valid/invalid configuration behavior as applicable.
- Use `pytest.approx()` for scalar floating-point comparisons and `torch.allclose()` for
  tensors. Check shape and device placement when relevant.
- Use the configured interpreter and commands in the [session workflow](../workflow/session.md).
  Scientific model changes also require validation-set evaluation before a PR when practical;
  report checks that were not run. Documentation-only changes do not require model evaluation.

## Hyperparameter Access

- Never use `self.hparams.get(...)`, `getattr(self.hparams, ..., default)`, or another fallback
  default for model or experiment hyperparameters.
- Declare every hyperparameter explicitly in `config/<model>_template.yml` (or the applicable required
  schema) and access it with `self.hparams.key` or `self.hparams["key"]`.
- A missing hyperparameter must fail during setup instead of silently changing model behavior.

