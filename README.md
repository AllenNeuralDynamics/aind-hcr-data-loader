# aind-hcr-data-loader

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-96.3%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-20%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Change log

**v0.6.0 (03/18/2026)**

*New modules*
+ Added `pairwise_dataset.py` —  `create_pairwise_unmixing_dataset()` for loading pairwise-unmixing pipeline outputs
+ Added `cell_typing_dataset.py` —  for attaching and querying cell-typing assets
+ Added `spot_filters.py` — spot-level filtering by channel-intensity percentiles, spatial isolation (kdtree), and spectral purity

*`hcr_dataset.py`*
+ Added `create_hcr_dataset_from_schema()` to construct an `HCRDataset` directly from an `ophys-mfish-dataset-catalog` JSON record (`mice/<mouse_id>.json`)
+ `create_hcr_dataset_from_config()` now auto-attaches cell-typing assets when a `"cell_typing"` key is present in the mouse config and accepts a new `metrics_base_path` parameter
+ `HCRRound` gains a `parent_dataset` reference (set automatically by `create_hcr_dataset`) enabling advanced filtering from round-level calls
+ `HCRRound.load_spots()` gains an `roi_filter_type` parameter (`'volume'` or `'comprehensive'`) as a cleaner alternative to passing raw `filter_cell_ids`
+ `HCRDataset` gains new methods: `get_filtered_cell_ids()`, `create_cell_gene_matrix_from_spots()`, `load_taxonomy_cell_types()`, `load_taxonomy_cell_types_h5ad()`, and `annotate_with_cell_types()`
+ `SpotFiles` replaces the `processing_manifest` field with `unmixed_cxg_filtered` and `mixed_cxg_filtered` for pairwise-unmixing filtered CxG tables
+ `get_spot_files()` now resolves pkl files using the round number from the key (e.g. `R2` → `mixed_spots_R2*.pkl`) to avoid accidentally picking up artefact files (e.g. `R-1`)
+ `get_processing_manifests()` now checks both `derived/processing_manifest.json` and the round root; raises `FileNotFoundError` with a clear message instead of an `AssertionError`
+ `create_channel_gene_table_from_manifests()` now adds a `round_channel_gene` convenience column (e.g. `"R2-488-GFP"`)
+ `create_channel_gene_table()` parameter renamed from `spot_files` → `processing_manifests` to reflect that it now takes manifest dicts directly

*`filters.py`*
+ `roi_filter_soma_and_overlap()` renamed to `roi_filter_comprehensive()` with an updated signature that no longer requires `HCRDataset` as a positional arg
+ `filter_tile_boundary_rois()` signature updated for consistency
+ Added `filter_cell_info()` — volume-quantile filter (default q1=0.2, q2=0.95) on a cell-info DataFrame
+ Added `get_inhibitory_mask()` — returns a boolean mask for inhibitory cells based on per-gene thresholds

*Dependencies*
+ Added `pyarrow` as a core dependency

**v0.5.1 02/23/2026**
+ Added metrics_base_path parameter to HCRDataset for soma shape classifier
+ Added new spot_filters.py module with vectorized percentile filtering and spot quality assessment functions
+ Enhanced filters.py with comprehensive ROI filtering pipeline including soma classification, edge detection, and tile overlap filtering
+ hcr_filters.ipynb notebook for filtering examples

**v0.4.0 (10/16/2025)**
+ extract tile overlap boundaries
+ adds ROI overlap calculation and filtering plus duplicate bbox plotting function.
+ added linear unmixing to single cell plots
+ new dye line plots
+ new pairwise intensity plots
+ added constants.py for channel colormaps
+ gather neuroglancer links
+ flexible figure saving via a new saveable_plot decorator

**v0.3.8 (8/11/2025)**
 + updated how cell info is gathered from segmentation sources (options for r1 centroids.npy file or union of multi-round mixed/unmixed cell_x_gene tables).
 + metrics path to segmentation files
 + better error handling when missing spot files

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

### Semantic Release

The table below, from [semantic release](https://github.com/semantic-release/semantic-release), shows which commit message gets you which release type when `semantic-release` runs (using the default configuration):

| Commit message                                                                                                                                                                                   | Release type                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fix(pencil): stop graphite breaking when too much pressure applied`                                                                                                                             | ~~Patch~~ Fix Release, Default release                                                                          |
| `feat(pencil): add 'graphiteWidth' option`                                                                                                                                                       | ~~Minor~~ Feature Release                                                                                       |
| `perf(pencil): remove graphiteWidth option`<br><br>`BREAKING CHANGE: The graphiteWidth option has been removed.`<br>`The default graphite width of 10mm is always used for performance reasons.` | ~~Major~~ Breaking Release <br /> (Note that the `BREAKING CHANGE: ` token must be in the footer of the commit) |

### Documentation
To generate the rst files source files for documentation, run
```bash
sphinx-apidoc -o docs/source/ src
```
Then to create the documentation HTML files, run
```bash
sphinx-build -b html docs/source/ docs/build/html
```
More info on sphinx installation can be found [here](https://www.sphinx-doc.org/en/master/usage/installation.html).

### Read the Docs Deployment
Note: Private repositories require **Read the Docs for Business** account. The following instructions are for a public repo.

The following are required to import and build documentations on *Read the Docs*:
- A *Read the Docs* user account connected to Github. See [here](https://docs.readthedocs.com/platform/stable/guides/connecting-git-account.html) for more details.
- *Read the Docs* needs elevated permissions to perform certain operations that ensure that the workflow is as smooth as possible, like installing webhooks. If you are not the owner of the repo, you may have to request elevated permissions from the owner/admin. 
- A **.readthedocs.yaml** file in the root directory of the repo. Here is a basic template:
```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Set the OS, Python version, and other tools you might need
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"

# Path to a Sphinx configuration file.
sphinx:
  configuration: docs/source/conf.py

# Declare the Python requirements required to build your documentation
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
```

Here are the steps for building docs in *Read the Docs*. See [here](https://docs.readthedocs.com/platform/stable/intro/add-project.html) for detailed instructions:
- From *Read the Docs* dashboard, click on **Add project**.
- For automatic configuration, select **Configure automatically** and type the name of the repo. A repo with public visibility should appear as you type. 
- Follow the subsequent steps.
- For manual configuration, select **Configure manually** and follow the subsequent steps

Once a project is created successfully, you will be able to configure/modify the project's settings; such as **Default version**, **Default branch** etc.
