"""CodeOcean utilities for attaching dataset-catalog records to capsules/pipelines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from codeocean import CodeOcean
from codeocean.data_asset import (
    DataAsset,
    DataAssetAttachParams,
    DataAssetAttachResults,
    DataAssetSearchParams,
)

logger = logging.getLogger(__name__)

# Environment variable names (matching the CodeOcean SDK convention)
_ENV_TOKEN = "API_SECRET"
_ENV_CAPSULE_ID = "CO_CAPSULE_ID"
_ENV_COMPUTATION_ID = "CO_COMPUTATION_ID"

# Allen Neural Dynamics CodeOcean domain — fixed for this organisation
CODEOCEAN_DOMAIN = "https://codeocean.allenneuraldynamics.org"


def create_client_from_env(token_env: str = _ENV_TOKEN) -> CodeOcean:
    """
    Create an authenticated :class:`~codeocean.CodeOcean` client.

    The domain is always ``https://codeocean.allenneuraldynamics.org``.
    The API token is read from the ``API_TOKEN`` environment variable,
    which CodeOcean injects automatically at capsule runtime.

    Parameters
    ----------
    token_env:
        Name of the environment variable holding the API token.
        Defaults to ``"API_TOKEN"``.

    Returns
    -------
    CodeOcean
        An authenticated client ready for use.

    Raises
    ------
    EnvironmentError
        If the token environment variable is unset or empty.

    Examples
    --------
    ::

        from aind_hcr_data_loader.codeocean_utils import create_client_from_env

        client = create_client_from_env()
    """
    token = os.environ.get(token_env, "")
    if not token:
        raise EnvironmentError(
            f"Missing required environment variable '{token_env}'. "
            "This is injected automatically by the CodeOcean runtime."
        )

    return CodeOcean(domain=CODEOCEAN_DOMAIN, token=token)


def get_capsule_id_from_env(capsule_id_env: str = _ENV_CAPSULE_ID) -> str:
    """
    Return the current capsule's ID from the ``CO_CAPSULE_ID`` environment
    variable, which CodeOcean injects automatically at runtime.

    Parameters
    ----------
    capsule_id_env:
        Name of the environment variable.  Defaults to ``"CO_CAPSULE_ID"``.

    Returns
    -------
    str
        The capsule UUID.

    Raises
    ------
    EnvironmentError
        If the environment variable is unset or empty.
    """
    capsule_id = os.environ.get(capsule_id_env, "")
    if not capsule_id:
        raise EnvironmentError(
            f"Environment variable '{capsule_id_env}' is not set. "
            "This variable is injected automatically by CodeOcean at capsule "
            "runtime. Are you running inside a capsule?"
        )
    return capsule_id


def get_computation_id_from_env(
    computation_id_env: str = _ENV_COMPUTATION_ID,
) -> str:
    """
    Return the current computation's ID from the ``CO_COMPUTATION_ID``
    environment variable, which CodeOcean injects automatically inside a
    cloud workstation session.

    Parameters
    ----------
    computation_id_env:
        Name of the environment variable.  Defaults to
        ``"CO_COMPUTATION_ID"``.

    Returns
    -------
    str
        The computation UUID.

    Raises
    ------
    EnvironmentError
        If the environment variable is unset or empty.
    """
    computation_id = os.environ.get(computation_id_env, "")
    if not computation_id:
        raise EnvironmentError(
            f"Environment variable '{computation_id_env}' is not set. "
            "This variable is injected automatically by CodeOcean inside a "
            "cloud workstation. Are you running inside a workstation?"
        )
    return computation_id


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MouseRecord:
    """
    Minimal representation of a mouse record from the ophys-mFISH dataset
    catalog (``mice/<mouse_id>.json``).

    Attributes
    ----------
    mouse_id : str
        The unique identifier for the mouse (e.g. ``"782149"``).
    rounds : dict[str, str]
        Mapping of round label → dataset asset name
        (e.g. ``{"R1": "HCR_782149_2025-11-05_..._processed_..."}``).
    derived_assets : dict[str, str]
        Mapping of derived-asset label → asset name
        (e.g. ``{"cell_typing": "HCR_782149_cell-typing_..."}``).
    schema_version : str
        Schema version of the catalog record.
    """

    mouse_id: str
    rounds: dict[str, str] = field(default_factory=dict)
    derived_assets: dict[str, str] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    @classmethod
    def from_dict(cls, data: dict) -> "MouseRecord":
        """Construct a :class:`MouseRecord` from a raw catalog JSON dict."""
        return cls(
            mouse_id=data["mouse_id"],
            rounds=data.get("rounds", {}),
            derived_assets=data.get("derived_assets", {}),
            schema_version=data.get("schema_version", "1.0.0"),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "MouseRecord":
        """Load a :class:`MouseRecord` from a catalog JSON file on disk."""
        import json

        with open(path) as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class AttachResult:
    """
    Result of attempting to attach a single named dataset asset to a capsule.

    Attributes
    ----------
    label : str
        Human-readable label from the catalog record
        (e.g. ``"rounds.R1"`` or ``"derived_assets.cell_typing"``).
    asset_name : str
        The CodeOcean data-asset name that was searched for.
    data_asset : DataAsset or None
        The resolved :class:`~codeocean.data_asset.DataAsset` object, or
        ``None`` if the asset could not be found.
    attach_results : list[DataAssetAttachResults]
        Raw attach results returned by the CodeOcean API. Empty list if the
        attach call was not made (e.g. asset not found or ``dry_run=True``).
    error : str or None
        Error message if something went wrong; ``None`` on success.
    """

    label: str
    asset_name: str
    data_asset: Optional[DataAsset] = None
    attach_results: list[DataAssetAttachResults] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """``True`` when the asset was found *and* attached without error."""
        return self.data_asset is not None and self.error is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_data_asset_by_name(
    client: CodeOcean,
    name: str,
) -> Optional[DataAsset]:
    """
    Search CodeOcean for a data asset whose name exactly matches *name*.

    Uses ``name:<name>`` query syntax supported by
    :class:`~codeocean.data_asset.DataAssetSearchParams`.  Returns the
    *first* matching asset, or ``None`` if nothing is found.

    Parameters
    ----------
    client:
        An authenticated :class:`~codeocean.CodeOcean` client.
    name:
        Exact data-asset name to search for.

    Returns
    -------
    DataAsset or None
    """
    search_params = DataAssetSearchParams(query=f'name:"{name}"')
    for asset in client.data_assets.search_data_assets_iterator(search_params):
        if asset.name == name:
            return asset
    return None


def print_attach_results(results: list[AttachResult], dry_run: bool = False) -> None:
    """
    Print a human-readable summary of :func:`attach_mouse_record*` results.

    Parameters
    ----------
    results:
        The list of :class:`AttachResult` objects returned by any
        ``attach_mouse_record_*`` function.
    dry_run:
        When ``True``, labels successful results as ``"found (dry run)"``
        instead of ``"attached"``.
    """
    for r in results:
        if r.success:
            status = "✓ found (dry run)" if dry_run else "✓ attached"
        else:
            status = f"✗ {r.error}"
        print(f"  [{r.label}]  {r.asset_name}  →  {status}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_mouse_record_to_capsule(
    record: MouseRecord,
    *,
    capsule_id: Optional[str] = None,
    client: Optional[CodeOcean] = None,
    include_rounds: bool = True,
    include_derived: bool = True,
    mount_prefix: str = "",
    dry_run: bool = False,
) -> list[AttachResult]:
    """
    Attach all round datasets and derived assets from a catalog record to a
    CodeOcean capsule.

    For every asset name listed in ``record.rounds`` (and/or
    ``record.derived_assets``), this function will:

    1. Search CodeOcean for a data asset whose name matches the catalog entry.
    2. Call :py:meth:`client.capsules.attach_data_assets` to attach it.

    Assets that cannot be found are reported in the returned
    :class:`AttachResult` objects with ``error`` set and are otherwise
    skipped — the function does *not* raise.

    Parameters
    ----------
    record:
        A :class:`MouseRecord` loaded from the dataset catalog.

    capsule_id:
        UUID of the target capsule.  If ``None`` (the default), the value is
        read from the ``CO_CAPSULE_ID`` environment variable, which CodeOcean
        injects automatically at capsule runtime.

    client:
        An authenticated :class:`~codeocean.CodeOcean` client.  If ``None``
        (the default), one is created automatically via
        :func:`create_client_from_env`, which reads the ``CODEOCEAN_URL`` and
        ``API_TOKEN`` environment variables.

    include_rounds:
        When ``True`` (default), attach all entries from ``record.rounds``.

    include_derived:
        When ``True`` (default), attach all entries from
        ``record.derived_assets``.

    mount_prefix:
        Optional string prepended to every mount path.  By default the
        data asset's own name is used as the mount path (CodeOcean default).
        If *mount_prefix* is non-empty the mount path becomes
        ``f"{mount_prefix}/{asset_name}"``.

    dry_run:
        When ``True``, perform the asset lookup but **skip** the actual
        attach API call.  Useful for validating that all assets can be
        resolved before committing.

    Returns
    -------
    list[AttachResult]
        One :class:`AttachResult` per catalog entry processed.  Check each
        entry's ``success`` property (or ``error``) to identify failures.

    Examples
    --------
    Fully automatic — inside a running CodeOcean capsule, just pass the record::

        from aind_hcr_data_loader.codeocean_utils import (
            MouseRecord,
            attach_mouse_record_to_capsule,
        )

        record = MouseRecord.from_json_file("mice/782149.json")

        # CO_CAPSULE_ID, CODEOCEAN_URL, and API_TOKEN are all injected
        # automatically by the CodeOcean runtime environment.
        results = attach_mouse_record_to_capsule(record)

        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"[{r.label}] {r.asset_name} -> {status}")

    Explicit overrides (e.g. targeting a different capsule from a notebook)::

        from codeocean import CodeOcean
        from aind_hcr_data_loader.codeocean_utils import (
            MouseRecord,
            attach_mouse_record_to_capsule,
        )

        client = CodeOcean(domain="https://acmecorp.codeocean.com", token="<token>")
        record = MouseRecord.from_json_file("mice/782149.json")

        results = attach_mouse_record_to_capsule(
            record,
            capsule_id="<your-capsule-uuid>",
            client=client,
        )
    """
    if client is None:
        client = create_client_from_env()
    if capsule_id is None:
        capsule_id = get_capsule_id_from_env()

    # Build the flat list of (label, asset_name) pairs to process
    entries: list[tuple[str, str]] = []

    if include_rounds:
        for round_label, asset_name in record.rounds.items():
            entries.append((f"rounds.{round_label}", asset_name))

    if include_derived:
        for derived_label, asset_name in record.derived_assets.items():
            entries.append((f"derived_assets.{derived_label}", asset_name))

    results: list[AttachResult] = []

    for label, asset_name in entries:
        result = AttachResult(label=label, asset_name=asset_name)

        # --- 1. Resolve the data asset by name ---
        logger.info(
            "mouse=%s  label=%s  Searching for data asset '%s'",
            record.mouse_id,
            label,
            asset_name,
        )
        try:
            data_asset = _find_data_asset_by_name(client, asset_name)
        except Exception as exc:
            result.error = f"Search failed: {exc}"
            logger.warning(
                "mouse=%s  label=%s  Search error: %s",
                record.mouse_id,
                label,
                exc,
            )
            results.append(result)
            continue

        if data_asset is None:
            result.error = f"Data asset not found in CodeOcean: '{asset_name}'"
            logger.warning(
                "mouse=%s  label=%s  %s",
                record.mouse_id,
                label,
                result.error,
            )
            results.append(result)
            continue

        result.data_asset = data_asset
        logger.info(
            "mouse=%s  label=%s  Found asset id=%s name='%s'",
            record.mouse_id,
            label,
            data_asset.id,
            data_asset.name,
        )

        # --- 2. Attach to capsule ---
        if dry_run:
            logger.info(
                "mouse=%s  label=%s  dry_run=True, skipping attach for asset id=%s",
                record.mouse_id,
                label,
                data_asset.id,
            )
            results.append(result)
            continue

        mount = (
            f"{mount_prefix}/{asset_name}" if mount_prefix else asset_name
        )
        attach_params = [DataAssetAttachParams(id=data_asset.id, mount=mount)]

        try:
            attach_results = client.capsules.attach_data_assets(
                capsule_id=capsule_id,
                attach_params=attach_params,
            )
            result.attach_results = attach_results
            logger.info(
                "mouse=%s  label=%s  Attached asset id=%s to capsule %s",
                record.mouse_id,
                label,
                data_asset.id,
                capsule_id,
            )
        except Exception as exc:
            result.error = f"Attach failed: {exc}"
            logger.error(
                "mouse=%s  label=%s  Attach error: %s",
                record.mouse_id,
                label,
                exc,
            )

        results.append(result)

    return results


def attach_mouse_record_to_pipeline(
    record: MouseRecord,
    *,
    pipeline_id: Optional[str] = None,
    client: Optional[CodeOcean] = None,
    include_rounds: bool = True,
    include_derived: bool = True,
    mount_prefix: str = "",
    dry_run: bool = False,
) -> list[AttachResult]:
    """
    Attach all round datasets and derived assets from a catalog record to a
    CodeOcean *pipeline*.

    Identical semantics to :func:`attach_mouse_record_to_capsule` but targets
    a pipeline instead of a capsule.

    Parameters
    ----------
    record:
        A :class:`MouseRecord` loaded from the dataset catalog.
    pipeline_id:
        UUID of the target pipeline.  If ``None`` (the default), the value is
        read from the ``CO_CAPSULE_ID`` environment variable.
        UUID of the target pipeline.
    record:
        A :class:`MouseRecord` loaded from the dataset catalog.
    client:
        An authenticated :class:`~codeocean.CodeOcean` client.  If ``None``
        (the default), one is created automatically via
        :func:`create_client_from_env`, which reads the ``CODEOCEAN_URL`` and
        ``API_TOKEN`` environment variables.
    include_rounds:
        When ``True`` (default), attach all entries from ``record.rounds``.
    include_derived:
        When ``True`` (default), attach all entries from
        ``record.derived_assets``.
    mount_prefix:
        Optional string prepended to every mount path.
    dry_run:
        When ``True``, resolve assets but skip the attach call.

    Returns
    -------
    list[AttachResult]
        One :class:`AttachResult` per catalog entry processed.
    """
    if client is None:
        client = create_client_from_env()
    if pipeline_id is None:
        pipeline_id = get_capsule_id_from_env()

    entries: list[tuple[str, str]] = []

    if include_rounds:
        for round_label, asset_name in record.rounds.items():
            entries.append((f"rounds.{round_label}", asset_name))

    if include_derived:
        for derived_label, asset_name in record.derived_assets.items():
            entries.append((f"derived_assets.{derived_label}", asset_name))

    results: list[AttachResult] = []

    for label, asset_name in entries:
        result = AttachResult(label=label, asset_name=asset_name)

        logger.info(
            "mouse=%s  label=%s  Searching for data asset '%s'",
            record.mouse_id,
            label,
            asset_name,
        )
        try:
            data_asset = _find_data_asset_by_name(client, asset_name)
        except Exception as exc:
            result.error = f"Search failed: {exc}"
            logger.warning(
                "mouse=%s  label=%s  Search error: %s",
                record.mouse_id,
                label,
                exc,
            )
            results.append(result)
            continue

        if data_asset is None:
            result.error = f"Data asset not found in CodeOcean: '{asset_name}'"
            logger.warning(
                "mouse=%s  label=%s  %s",
                record.mouse_id,
                label,
                result.error,
            )
            results.append(result)
            continue

        result.data_asset = data_asset
        logger.info(
            "mouse=%s  label=%s  Found asset id=%s name='%s'",
            record.mouse_id,
            label,
            data_asset.id,
            data_asset.name,
        )

        if dry_run:
            logger.info(
                "mouse=%s  label=%s  dry_run=True, skipping attach for asset id=%s",
                record.mouse_id,
                label,
                data_asset.id,
            )
            results.append(result)
            continue

        mount = (
            f"{mount_prefix}/{asset_name}" if mount_prefix else asset_name
        )
        attach_params = [DataAssetAttachParams(id=data_asset.id, mount=mount)]

        try:
            attach_results = client.pipelines.attach_data_assets(
                pipeline_id=pipeline_id,
                attach_params=attach_params,
            )
            result.attach_results = attach_results
            logger.info(
                "mouse=%s  label=%s  Attached asset id=%s to pipeline %s",
                record.mouse_id,
                label,
                data_asset.id,
                pipeline_id,
            )
        except Exception as exc:
            result.error = f"Attach failed: {exc}"
            logger.error(
                "mouse=%s  label=%s  Attach error: %s",
                record.mouse_id,
                label,
                exc,
            )

        results.append(result)

    return results


def attach_mouse_record_to_workstation(
    record: MouseRecord,
    *,
    computation_id: Optional[str] = None,
    client: Optional[CodeOcean] = None,
    include_rounds: bool = True,
    include_derived: bool = True,
    mount_prefix: str = "",
    dry_run: bool = False,
) -> list[AttachResult]:
    """
    Attach all round datasets and derived assets from a catalog record to the
    current CodeOcean *cloud workstation* computation.

    This uses ``client.computations.attach_data_assets``, which is the correct
    API when running inside a cloud workstation session (as opposed to a
    capsule or pipeline, which use ``client.capsules`` / ``client.pipelines``).

    Parameters
    ----------
    record:
        A :class:`MouseRecord` loaded from the dataset catalog.
    computation_id:
        UUID of the current workstation computation.  If ``None`` (the
        default), the value is read from the ``CO_COMPUTATION_ID`` environment
        variable, which CodeOcean injects automatically inside a cloud
        workstation.
    client:
        An authenticated :class:`~codeocean.CodeOcean` client.  If ``None``
        (the default), one is created automatically via
        :func:`create_client_from_env`.
    include_rounds:
        When ``True`` (default), attach all entries from ``record.rounds``.
    include_derived:
        When ``True`` (default), attach all entries from
        ``record.derived_assets``.
    mount_prefix:
        Optional string prepended to every mount path.  By default the
        data asset's own name is used as the mount path.  If *mount_prefix*
        is non-empty the mount path becomes
        ``f"{mount_prefix}/{asset_name}"``.
    dry_run:
        When ``True``, perform the asset lookup but **skip** the actual
        attach API call.  Useful for validating that all assets can be
        resolved before committing.

    Returns
    -------
    list[AttachResult]
        One :class:`AttachResult` per catalog entry processed.

    Examples
    --------
    ::

        from aind_hcr_data_loader.codeocean_utils import (
            MouseRecord,
            attach_mouse_record_to_workstation,
        )

        record = MouseRecord.from_json_file("mice/782149.json")

        # CO_COMPUTATION_ID and API_SECRET are injected automatically by the
        # CodeOcean cloud workstation environment.
        results = attach_mouse_record_to_workstation(record)

        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"[{r.label}] {r.asset_name} -> {status}")
    """
    if client is None:
        client = create_client_from_env()
    if computation_id is None:
        computation_id = get_computation_id_from_env()

    entries: list[tuple[str, str]] = []

    if include_rounds:
        for round_label, asset_name in record.rounds.items():
            entries.append((f"rounds.{round_label}", asset_name))

    if include_derived:
        for derived_label, asset_name in record.derived_assets.items():
            entries.append((f"derived_assets.{derived_label}", asset_name))

    results: list[AttachResult] = []

    for label, asset_name in entries:
        result = AttachResult(label=label, asset_name=asset_name)

        logger.info(
            "mouse=%s  label=%s  Searching for data asset '%s'",
            record.mouse_id,
            label,
            asset_name,
        )
        try:
            data_asset = _find_data_asset_by_name(client, asset_name)
        except Exception as exc:
            result.error = f"Search failed: {exc}"
            logger.warning(
                "mouse=%s  label=%s  Search error: %s",
                record.mouse_id,
                label,
                exc,
            )
            results.append(result)
            continue

        if data_asset is None:
            result.error = f"Data asset not found in CodeOcean: '{asset_name}'"
            logger.warning(
                "mouse=%s  label=%s  %s",
                record.mouse_id,
                label,
                result.error,
            )
            results.append(result)
            continue

        result.data_asset = data_asset
        logger.info(
            "mouse=%s  label=%s  Found asset id=%s name='%s'",
            record.mouse_id,
            label,
            data_asset.id,
            data_asset.name,
        )

        if dry_run:
            logger.info(
                "mouse=%s  label=%s  dry_run=True, skipping attach for asset id=%s",
                record.mouse_id,
                label,
                data_asset.id,
            )
            results.append(result)
            continue

        mount = (
            f"{mount_prefix}/{asset_name}" if mount_prefix else asset_name
        )
        attach_params = [DataAssetAttachParams(id=data_asset.id, mount=mount)]

        try:
            attach_results = client.computations.attach_data_assets(
                computation_id=computation_id,
                attach_params=attach_params,
            )
            result.attach_results = attach_results
            logger.info(
                "mouse=%s  label=%s  Attached asset id=%s to workstation computation %s",
                record.mouse_id,
                label,
                data_asset.id,
                computation_id,
            )
        except Exception as exc:
            result.error = f"Attach failed: {exc}"
            logger.error(
                "mouse=%s  label=%s  Attach error: %s",
                record.mouse_id,
                label,
                exc,
            )

        results.append(result)

    return results


def attach_mouse_record(
    record: MouseRecord,
    *,
    capsule_id: Optional[str] = None,
    computation_id: Optional[str] = None,
    pipeline_id: Optional[str] = None,
    client: Optional[CodeOcean] = None,
    include_rounds: bool = True,
    include_derived: bool = True,
    mount_prefix: str = "",
    dry_run: bool = False,
) -> list[AttachResult]:
    """
    Convenience dispatcher: attach a catalog record to whatever CodeOcean
    context is currently active.

    Detects the runtime context by inspecting environment variables in the
    following priority order:

    1. ``CO_COMPUTATION_ID`` is set (or *computation_id* is supplied)
       → calls :func:`attach_mouse_record_to_workstation`.
    2. ``CO_CAPSULE_ID`` is set (or *capsule_id* is supplied)
       → calls :func:`attach_mouse_record_to_capsule`.
    3. *pipeline_id* is supplied explicitly
       → calls :func:`attach_mouse_record_to_pipeline`.

    If none of the above conditions can be satisfied, an
    :class:`EnvironmentError` is raised.

    Parameters
    ----------
    record:
        A :class:`MouseRecord` loaded from the dataset catalog.
    capsule_id:
        Explicit capsule UUID.  Takes effect only when no workstation context
        is detected.  When ``None``, ``CO_CAPSULE_ID`` is consulted.
    computation_id:
        Explicit workstation computation UUID.  When supplied, always triggers
        the workstation path regardless of other env vars.  When ``None``,
        ``CO_COMPUTATION_ID`` is consulted.
    pipeline_id:
        Explicit pipeline UUID.  Only used when neither a workstation nor
        capsule context is detected/supplied.
    client:
        An authenticated :class:`~codeocean.CodeOcean` client.  If ``None``
        (the default), one is created automatically via
        :func:`create_client_from_env`.
    include_rounds:
        When ``True`` (default), attach all entries from ``record.rounds``.
    include_derived:
        When ``True`` (default), attach all entries from
        ``record.derived_assets``.
    mount_prefix:
        Optional string prepended to every mount path.
    dry_run:
        When ``True``, resolve assets but skip the attach call.

    Returns
    -------
    list[AttachResult]
        One :class:`AttachResult` per catalog entry processed.

    Raises
    ------
    EnvironmentError
        If the runtime context cannot be determined from environment variables
        and no explicit IDs are provided.

    Examples
    --------
    Inside any CodeOcean runtime — workstation, capsule, or pipeline — just
    pass the record and let the dispatcher figure out the rest::

        from aind_hcr_data_loader.codeocean_utils import (
            MouseRecord,
            attach_mouse_record,
        )

        record = MouseRecord.from_json_file("mice/782149.json")
        results = attach_mouse_record(record)

        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"[{r.label}] {r.asset_name} -> {status}")
    """
    _shared = dict(
        client=client,
        include_rounds=include_rounds,
        include_derived=include_derived,
        mount_prefix=mount_prefix,
        dry_run=dry_run,
    )

    # 1. Workstation: explicit argument or env var present
    resolved_computation_id = computation_id or os.environ.get(
        _ENV_COMPUTATION_ID, ""
    )
    if resolved_computation_id:
        logger.info(
            "mouse=%s  dispatch=workstation  computation_id=%s",
            record.mouse_id,
            resolved_computation_id,
        )
        return attach_mouse_record_to_workstation(
            record, computation_id=resolved_computation_id, **_shared
        )

    # 2. Capsule: explicit argument or env var present
    resolved_capsule_id = capsule_id or os.environ.get(_ENV_CAPSULE_ID, "")
    if resolved_capsule_id:
        logger.info(
            "mouse=%s  dispatch=capsule  capsule_id=%s",
            record.mouse_id,
            resolved_capsule_id,
        )
        return attach_mouse_record_to_capsule(
            record, capsule_id=resolved_capsule_id, **_shared
        )

    # 3. Pipeline: explicit argument only (no standard env var)
    if pipeline_id:
        logger.info(
            "mouse=%s  dispatch=pipeline  pipeline_id=%s",
            record.mouse_id,
            pipeline_id,
        )
        return attach_mouse_record_to_pipeline(
            record, pipeline_id=pipeline_id, **_shared
        )

    raise EnvironmentError(
        "Cannot determine CodeOcean runtime context. "
        f"Set '{_ENV_COMPUTATION_ID}' (workstation), '{_ENV_CAPSULE_ID}' "
        "(capsule), or pass 'pipeline_id' explicitly."
    )
