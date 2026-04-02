# aind-hcr-data-loader

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-96.3%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-20%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Change log

See [CHANGELOG.md](CHANGELOG.md) for full details.

| Version | Date | Summary |
|---------|------|---------|
| v0.7.0 | 04/02/2026 | New `codeocean_utils` module for attaching dataset-catalog records to capsules/pipelines |
| v0.6.0 | 03/18/2026 | New pairwise/cell-typing/spot-filter modules; major `HCRDataset` and `filters.py` additions |
| v0.5.1 | 02/23/2026 | `metrics_base_path`, `spot_filters.py`, comprehensive ROI filtering pipeline |
| v0.4.0 | 10/16/2025 | Tile overlap, ROI filtering, linear unmixing, neuroglancer links, `saveable_plot` |
| v0.3.8 | 08/11/2025 | Cell info loading improvements, metrics paths, error handling |

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```bash
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```bash
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```bash
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```bash
black .
```

- Use **isort** to automatically sort import statements:
```bash
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests