# S3-Native Dataset Loading — Problem Analysis & Design Proposal

## Background

`create_hcr_dataset_from_s3` was added to allow loading HCR datasets directly from S3
without requiring a CodeOcean workstation or locally mounted data assets. The entry point
sets `data_dir = Path("s3://aind-open-data")` and delegates to the existing
`create_hcr_dataset_from_schema` → `create_hcr_dataset` call chain.

---

## Problem

### Root cause

`pathlib.Path` silently mangles S3 URIs:

```python
Path("s3://bucket") / "folder"
# → PosixPath("s3:/bucket/folder")   ← one slash eaten by POSIX normalisation
```

The resulting path looks like a local absolute path to the OS and every filesystem
call fails with `FileNotFoundError` or `OSError`.

### Scope of impact

The entire file-discovery and file-reading stack assumes local filesystem semantics:

| Layer | Calls used | Works on S3? |
|---|---|---|
| `get_processing_manifests` | `folder_path.iterdir()` | No |
| `get_spot_detection_files` | `path.exists()`, `path.iterdir()`, `path.glob()` | No |
| `get_segmentation_files` | `path.exists()` | No |
| `get_tile_alignment_files` | `path.exists()` | No |
| `get_metadata_files` | `path.exists()` | No |
| `get_spot_files` | `path.exists()`, `path.glob()`, `path.absolute()` | No |
| `get_zarr_files` | `path.exists()`, `path.glob()` | No |
| `HCRRound.load_spots` | `open(path, "rb")` | No |
| `HCRRound.get_cell_centroids` | `np.load(path)` | No |
| `HCRRound.get_metadata` | `open(path, "r")` | No |
| `load_round_spots` (standalone) | `path.exists()`, `open()` | No |
| `load_processing_manifest` | `open()` | No |
| `zarr.open(zarr_path)` | native zarr | Partially — zarr supports S3 with `s3://` strings but not `Path` objects |

### What does work on S3

- `pd.read_csv("s3://...")` — pandas has built-in S3 support via fsspec
- `zarr.open("s3://...", mode="r")` — zarr supports S3 strings natively
- `fsspec.open("s3://...", "rb")` — explicit fsspec open works for any file

---

## Design Options

### Option A — `cloudpathlib` (not currently installed)

Replace `pathlib.Path` with `cloudpathlib.S3Path` throughout. Preserves the
`path / "sub" / "file.csv"` ergonomics and `.exists()`, `.glob()`, `.iterdir()` work
transparently. **Requires adding `cloudpathlib[s3]` as a dependency.**  This is the
cleanest long-term solution but touches many call sites.

### Option B — `upath.UPath` / `universal_pathlib` (not currently installed)

Same idea as cloudpathlib, different library. Also requires a new dependency.

### Option C — `fsspec` AbstractFileSystem (already installed)

Introduce a thin `_fs_path` abstraction: a helper that, given a root (local `Path` or
`"s3://bucket"`), returns an `(fs, base_str)` pair where `fs` is an `fsspec`
filesystem object. All `get_*` functions and file readers receive this pair and call
`fs.exists(path)`, `fs.ls(path)`, `fs.glob(path)`, `fs.open(path)` instead of
pathlib methods. No new dependencies. More verbose at call sites but fully explicit.

### Option D — Pre-build path strings, skip discovery on S3 (pragmatic shortcut)

Since the catalog schema already contains all asset names and the directory structure
inside each asset is deterministic (fixed subfolder names like
`image_spot_spectral_unmixing/`, `cell_body_segmentation/`, etc.), the `get_*`
functions could be rewritten to **build paths from known templates** and skip
`.exists()` / `.iterdir()` checks entirely when `data_dir` is an S3 URI.  Path
objects are only used for local existence checks; on S3, the path is recorded as a
string and checked lazily on first read (let the reader raise, not the constructor).
This is the **smallest change surface** but only safe because the pipeline output
structure is stable.

---

## Recommendation

**Short term — Option D**, because:

- No new dependencies
- Confined to the `get_*` constructor helpers and `create_hcr_dataset_from_s3`
- The directory layout is stable and owned by the same codebase
- File readers already support `fsspec`-compatible strings for CSV (`pd.read_csv`)
  and zarr (`zarr.open`) — only `open()` and `np.load()` need wrapping with
  `fsspec.open()`

**Longer term — Option A** (`cloudpathlib`), once the dependency is approved.
It makes the whole codebase location-agnostic without special-casing S3 everywhere.

---

## Proposed Short-term Implementation Plan (Option D)

### 1. Add `_is_s3(path) -> bool` utility

```python
def _is_s3(path) -> bool:
    return str(path).startswith("s3://")
```

### 2. Add `_s3_join(*parts) -> str` utility

Constructs an S3 URI by joining with `/`, without touching `pathlib.Path`:

```python
def _s3_join(*parts) -> str:
    return "/".join(str(p).rstrip("/") for p in parts)
```

### 3. Refactor `get_*` functions

Each function checks `_is_s3(data_dir)`:

- **Local path**: current behaviour unchanged — use `pathlib.Path` with `.exists()` checks
- **S3 path**: build path strings using `_s3_join`, skip `.exists()` / `.iterdir()` /
  `.glob()` checks, store paths as strings (or lightweight wrappers). Optional files
  (tiles, metadata, etc.) are set to `None` by default on S3 since existence can't be
  checked cheaply without a network call; callers already guard against `None`.

For `get_processing_manifests` specifically, use `fsspec` to fetch the manifest JSON
directly from the known S3 path rather than scanning the directory.

For `get_spot_files`, the round number in the filename is deterministic from the
round key — no glob needed.

### 4. Wrap file readers with `fsspec.open`

Replace the three non-fsspec-aware read patterns:

| Current | Replacement |
|---|---|
| `open(path, "rb")` → `pd.read_pickle` | `fsspec.open(path, "rb")` |
| `open(path, "rb")` → `pkl.load` | `fsspec.open(path, "rb")` |
| `open(path, "r")` → `json.load` | `fsspec.open(path, "r")` |
| `np.load(path)` | `fsspec.open(path, "rb")` → `np.load(f)` |

`pd.read_csv` and `zarr.open` already accept S3 strings natively — no change needed.

### 5. `create_hcr_dataset_from_s3` passes `data_dir` as a string, not a `Path`

The `Path(f"s3://{bucket}")` construction is the immediate bug source. Change to:

```python
data_dir = f"s3://{bucket}"
```

and propagate `str`-typed `data_dir` through `create_hcr_dataset_from_schema` and
`create_hcr_dataset` (currently typed as `Path` — relax to `Union[str, Path]`).

---

## Files to change

| File | Change |
|---|---|
| `hcr_dataset.py` | `_is_s3`, `_s3_join` utilities; refactor all `get_*` and reader methods |
| `pairwise_dataset.py` | Check for same `open()` / `np.load()` patterns |
| `hcr_dataset.py` → `create_hcr_dataset_from_s3` | Pass `str` not `Path` for S3 data_dir |

---

## Out of scope

- Writing back to S3
- Streaming zarr data differently (zarr already handles this)
- Supporting GCS or Azure (fsspec supports both; the abstraction generalises if needed)
