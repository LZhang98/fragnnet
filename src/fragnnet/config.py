"""Run configuration: the shared template, an experiment config, CLI overrides and W&B values.

Every model reads the same ``config/template.yml``, so shared code can read any key without
checking ``model_type`` first, and a key the template does not declare is a typo rather than
another model's setting. Every source of config values (template, experiment config, override,
saved W&B config) goes through the same migrations before it is merged.
"""

import logging
import os

import yaml

from fragnnet import registry
from fragnnet.utils.migrations import (
    RETIRED_INST_TYPE_HELP,
    find_retired_inst_type_refs,
    migrate_legacy_masster_name,
    migrate_masster_proposal_method,
    retire_pair_bce_hparams,
)
from fragnnet.utils.misc_utils import deep_update

logger = logging.getLogger(__name__)

# The registry's set only guards against a typo in `model_type`.
VALID_MODEL_TYPES = registry.VALID_MODEL_TYPES
TEMPLATE_FILENAME = "template.yml"
# What a config that declares no `model_type` means, and what `config/template.yml` declares.
# Only reached by a template that omits the key, e.g. a minimal one saved beside a checkpoint.
DEFAULT_MODEL_TYPE = "frag_gnn"

# Mappings whose sub-keys are data (instrument names, split names), not schema. Custom
# configs may introduce keys inside them, so they are not validated recursively.
OPEN_CONFIG_MAPPINGS = frozenset(
    {
        "spec_params.merge_tolerance_by_inst",
        "spec_params.split_filters",
        "ce_context_ce_key_by_inst",
        "inst_type_loss_tol",
    }
)


def migrate_config_dict(config_d: dict) -> None:
    """Bring a config mapping from any source up to the current key names, in place.

    Drops disabled retired pair-BCE settings (and rejects enabled ones), renames pre-rename
    MASSter keys and the original proposal key. All three act on top-level keys only.

    Args:
        config_d: A template, experiment config, override or saved W&B config.

    Raises:
        ValueError: If the mapping enables a retired objective.
    """
    retire_pair_bce_hparams(config_d)
    migrate_legacy_masster_name(config_d)
    migrate_masster_proposal_method(config_d)


def raise_if_retired_inst_types(config_d: dict, source: str) -> None:
    """Refuse a config that names a retired instrument type ("FT", "OB", "OB_F").

    Relabelled data has none, so a table or list still keyed by one would silently stop
    applying to Orbitrap spectra.

    Args:
        config_d: Resolved config.
        source: What produced the config, for the error message.

    Raises:
        ValueError: If any key, list item or string value is a retired instrument type.
    """
    retired = find_retired_inst_type_refs(config_d)
    if retired:
        raise ValueError(
            f"{source} names a retired instrument type at {retired}. {RETIRED_INST_TYPE_HELP}"
        )


def resolve_template_path(template_path: str) -> str:
    """Resolve the config template.

    ``template_path`` is normally the config directory holding ``template.yml``. A path to a
    file is used as-is, which is how a template saved inside a checkpoint still resolves.

    Args:
        template_path: Config directory, or a single template file.

    Returns:
        Path to the template to load.

    Raises:
        FileNotFoundError: If no template exists at that path.
    """
    if os.path.isfile(template_path):
        return template_path
    if not os.path.isdir(template_path):
        raise FileNotFoundError(f"no config template at {template_path}")
    resolved_fp = os.path.join(template_path, TEMPLATE_FILENAME)
    if not os.path.isfile(resolved_fp):
        raise FileNotFoundError(f"missing config template: {resolved_fp}")
    return resolved_fp


def find_unknown_config_keys(custom_d: dict, config_d: dict, prefix: str = "") -> list[str]:
    """Collect dotted paths in ``custom_d`` that the resolved config does not declare.

    Recurses into nested sections so a typo inside ``spec_params`` is caught as loudly as one
    at the top level. Sections listed in :data:`OPEN_CONFIG_MAPPINGS` hold data rather than a
    schema and are not descended into.

    Args:
        custom_d: The custom config being validated.
        config_d: The resolved model template.
        prefix: Dotted path of the section currently being checked.

    Returns:
        Sorted dotted paths that are not declared by the resolved config.

    Raises:
        ValueError: If a declared config section is replaced with a non-mapping.
    """
    unknown = []
    for key, value in custom_d.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in config_d:
            unknown.append(path)
            continue
        if (
            path not in OPEN_CONFIG_MAPPINGS
            and isinstance(config_d[key], dict)
            and not isinstance(value, dict)
        ):
            raise ValueError(f"config section {path!r} must be a mapping; got a non-mapping")
        if (
            path not in OPEN_CONFIG_MAPPINGS
            and isinstance(value, dict)
            and isinstance(config_d[key], dict)
        ):
            unknown.extend(find_unknown_config_keys(value, config_d[key], path))
    return sorted(unknown)


def load_custom_config(custom_fp: str) -> dict:
    """Read and migrate only the explicitly declared experiment or overlay settings."""
    if not os.path.isfile(custom_fp):
        raise FileNotFoundError(f"custom config not found: {custom_fp}")
    with open(custom_fp) as custom_file:
        custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
    if custom_d is None:
        custom_d = {}
    if not isinstance(custom_d, dict):
        raise ValueError(f"{custom_fp} must contain a config mapping")
    migrate_config_dict(custom_d)
    return custom_d


def apply_override(config_d: dict, override_item: str) -> None:
    """Apply one ``key=value`` CLI override to ``config_d``, in place.

    The value is parsed as YAML and dotted keys address nested sections. The key goes through
    :func:`migrate_config_dict` like a config file, so a legacy spelling still applies and a
    disabled retired setting is dropped.
    Mapping values merge recursively with the existing section after schema validation.

    Args:
        config_d: Resolved config.
        override_item: ``key=value``; e.g. ``spec_params.nce=False``.

    Raises:
        ValueError: If the item is not ``key=value``, a section on the path does not exist, or
            the key is undeclared outside :data:`OPEN_CONFIG_MAPPINGS`.
    """
    if "=" not in override_item:
        raise ValueError(f"override {override_item!r} is not key=value")
    dotted_key, raw_value = override_item.split("=", 1)
    top_key, *sub_keys = dotted_key.split(".")
    # 1. Migrate the override as the one-key config it is, then read the value back out.
    nested = yaml.safe_load(raw_value)
    for sub_key in reversed(sub_keys):
        nested = {sub_key: nested}
    migrated = {top_key: nested}
    migrate_config_dict(migrated)
    for migrated_key, value in migrated.items():
        for sub_key in sub_keys:
            value = value[sub_key]
        path = [migrated_key, *sub_keys]
        # 2. Walk to the section that holds the key; every section must exist.
        section = config_d
        for depth, section_key in enumerate(path[:-1]):
            if not isinstance(section.get(section_key), dict):
                raise ValueError(
                    f"override {dotted_key!r}: {'.'.join(path[: depth + 1])} is not a "
                    "config section"
                )
            section = section[section_key]
        if path[-1] not in section and ".".join(path[:-1]) not in OPEN_CONFIG_MAPPINGS:
            raise ValueError(f"override {dotted_key!r} sets a key the template does not declare")
        parent_path = ".".join(path[:-1])
        if parent_path not in OPEN_CONFIG_MAPPINGS:
            unknown = find_unknown_config_keys({path[-1]: value}, section, parent_path)
            if unknown:
                raise ValueError(f"override sets keys the template does not declare: {unknown}")
        logger.info("Updating %s from %s to %s", ".".join(path), section.get(path[-1]), value)
        if isinstance(section.get(path[-1]), dict) and isinstance(value, dict):
            section[path[-1]] = deep_update(section[path[-1]], value)
        else:
            section[path[-1]] = value


def load_config(
    template_path: str,
    custom_fp: str | None,
    override: list | None,
    model_type: str | None = None,
) -> dict:
    """Resolve the config template, the experiment config, and CLI overrides.

    Every model reads the same ``config/template.yml``. Passing a single template file instead
    of the directory uses that file directly, which is how a template saved inside a checkpoint
    resolves.

    ``model_type`` comes from a CLI override, else the custom config, else the ``model_type``
    argument (the type recorded in a checkpoint), else the template.

    Args:
        template_path: Config directory, or a single template file.
        custom_fp: Path to the experiment config YAML, or None.
        override: ``key=value`` strings; dotted keys address nested sections.
        model_type: Model type to fall back on when the custom config does not declare one,
            e.g. the type recorded in a checkpoint.

    Returns:
        The resolved configuration dictionary.

    Raises:
        FileNotFoundError: If the template or the custom config is missing.
        ValueError: If ``model_type`` is unknown or retired, the custom config or an override
            sets undeclared keys, or the result names a retired instrument type.
    """
    # 1. The experiment config.
    custom_d = None
    if custom_fp is not None:
        custom_d = load_custom_config(custom_fp)

    # 2. The template every model shares. One saved inside a pre-rename checkpoint still
    # spells MASSter `mass_shatter`.
    with open(resolve_template_path(template_path)) as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)
    migrate_config_dict(config_d)
    # A caller-supplied model_type (a checkpoint's) outranks the template default, so an
    # inference config that stays silent about the model does not silently become frag_gnn.
    resolved_model_type = (
        (custom_d or {}).get("model_type")
        or model_type
        or config_d.get("model_type")
        or DEFAULT_MODEL_TYPE
    )
    selection = {"model_type": resolved_model_type}
    for override_item in override or []:
        if override_item.split("=", 1)[0] == "model_type":
            apply_override(selection, override_item)
    resolved_model_type = selection["model_type"]
    if not isinstance(resolved_model_type, str):
        raise ValueError(f"unknown model_type {resolved_model_type!r}; expected a string")
    registry.raise_if_retired(resolved_model_type)
    if resolved_model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"unknown model_type {resolved_model_type!r}; known types: {sorted(VALID_MODEL_TYPES)}"
        )
    config_d["model_type"] = resolved_model_type
    registry.get_model_hooks(resolved_model_type).migrate_template_config(config_d)

    # 3. Validate and merge the experiment config. Validation recurses into nested sections,
    # so a typo inside `spec_params` fails as loudly as one at the top.
    if custom_d is not None:
        unknown = find_unknown_config_keys(custom_d, config_d)
        if unknown:
            raise ValueError(f"{custom_fp} sets keys the template does not declare: {unknown}")
        config_d = deep_update(config_d, custom_d)
    config_d["model_type"] = resolved_model_type

    # 4. CLI overrides, then the retired-instrument check over the result.
    for override_item in override or []:
        if override_item.split("=", 1)[0] != "model_type":
            apply_override(config_d, override_item)
    raise_if_retired_inst_types(config_d, "config")
    return config_d


def merge_wandb_config(config_d: dict, saved_run_config: dict, *, resumed: bool = False) -> None:
    """Take the values W&B holds for this run (sweep or resumed run) into ``config_d``, in place.

    Nested mappings merge recursively. A value is replaced only when it differs: W&B stores
    config through JSON and returns an integral float as an int (``1.0`` comes back as ``1``),
    which is equal but would rename the metrics built from it, e.g. ``cos_sim_1.0`` to
    ``cos_sim_1``, away from ``checkpoint_metric``.

    On a new run, a key ``config_d`` does not declare can only come from the sweep, so it is a
    typo and raises, as it does in a custom config. A resumed run's saved config may hold keys
    the template has since dropped; those are logged and skipped.

    Args:
        config_d: Resolved run configuration, updated in place.
        saved_run_config: ``dict(wandb.config)`` after ``wandb.init``; not modified.
        resumed: Whether ``saved_run_config`` was saved by an earlier attempt of this job.

    Raises:
        ValueError: If a new run's W&B config sets undeclared keys, a section is not a mapping,
            the merge would change ``model_type``, or the merged config names a retired
            instrument type.
    """
    saved_run_config = dict(saved_run_config)
    migrate_config_dict(saved_run_config)
    unknown = find_unknown_config_keys(saved_run_config, config_d)
    if unknown and not resumed:
        raise ValueError(f"W&B config sets keys the template does not declare: {unknown}")
    if unknown:
        logger.warning("Skipping keys the template no longer declares: %s", unknown)
    _merge_wandb_mapping(config_d, saved_run_config)
    raise_if_retired_inst_types(config_d, "W&B config")


def _merge_wandb_mapping(
    config_d: dict, saved_config: dict, prefix: str = "", open_mapping: bool = False
) -> None:
    for key, value in saved_config.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in config_d:
            if open_mapping:
                config_d[key] = value
            continue
        if path == "model_type" and config_d[key] != value:
            raise ValueError(
                f"W&B config sets model_type {value!r}; the config resolved {config_d[key]!r}"
            )
        if isinstance(config_d[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"W&B config sets section {path!r} to a non-mapping")
            _merge_wandb_mapping(
                config_d[key], value, path, open_mapping or path in OPEN_CONFIG_MAPPINGS
            )
        elif config_d[key] != value:
            logger.info("config diff -- %s: %s vs %s", path, config_d[key], value)
            config_d[key] = value
