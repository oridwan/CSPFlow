"""Layered configuration loading.

    defaults.yaml  <  machines/<site>.yaml  <  templates/<app>.yaml
                   <  campaign.yaml  <  CLI --set

Every leaf value remembers which layer supplied it, so ``csp config show
--resolved`` can answer "where does this setting come from" without anyone
having to reason about merge order.  That question being unanswerable is one of
the concrete complaints in pipeline.md sec.1.2.

A machine profile is mostly a `Machine` document, but it may also carry a
``campaign:`` block of site-specific campaign defaults; only that block joins
the campaign layer chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import Campaign, Machine

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = PACKAGE_ROOT / "config" / "defaults.yaml"
MACHINES_DIR = PACKAGE_ROOT / "machines"
TEMPLATES_DIR = PACKAGE_ROOT / "templates"

_VAR = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


class ConfigError(Exception):
    """Raised for anything wrong with the configuration, with a usable message."""


# --------------------------------------------------------------------------
# variable expansion
# --------------------------------------------------------------------------


def expand_vars(obj: Any, env: dict[str, str] | None = None, *, _path: str = "") -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` in strings.

    An undefined variable is an error rather than an empty string: silently
    expanding ``$SCRATCH`` to ``""`` turns ``$SCRATCH/campaign`` into an
    absolute path at the filesystem root, which is the kind of quiet mistake
    this pipeline exists to eliminate.
    """
    src = os.environ if env is None else env

    if isinstance(obj, dict):
        return {k: expand_vars(v, env, _path=f"{_path}.{k}" if _path else k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_vars(v, env, _path=f"{_path}[{i}]") for i, v in enumerate(obj)]
    if not isinstance(obj, str):
        return obj

    missing: list[str] = []

    def repl(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name not in src:
            missing.append(name)
            return m.group(0)
        return str(src[name])

    out = _VAR.sub(repl, obj)
    if missing:
        where = f" (at {_path})" if _path else ""
        raise ConfigError(
            f"undefined variable(s) {sorted(set(missing))} in {obj!r}{where}. "
            f"Export them or set the value explicitly."
        )
    return out


# --------------------------------------------------------------------------
# merging with provenance
# --------------------------------------------------------------------------


def deep_merge(
    base: dict[str, Any],
    over: dict[str, Any],
    layer: str,
    origins: dict[str, str],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Merge ``over`` onto ``base``, recording the origin layer of each leaf.

    Lists replace wholesale rather than concatenating.  Appending would make
    ``source:`` and ``properties:`` impossible to shorten from a later layer,
    and "why is this element still here" is a worse question than "why did this
    list get replaced".
    """
    out = dict(base)
    for key, value in over.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value, layer, origins, prefix=path)
        else:
            out[key] = value
            origins[path] = layer
            if isinstance(value, dict):
                _mark_subtree(value, layer, origins, path)
    return out


def _mark_subtree(node: Any, layer: str, origins: dict[str, str], prefix: str) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}.{k}"
            origins[p] = layer
            _mark_subtree(v, layer, origins, p)


def parse_set(assignment: str) -> tuple[str, Any]:
    """Parse one ``--set a.b.c=value``.  Values are parsed as YAML scalars."""
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, _, raw = assignment.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"--set has an empty key: {assignment!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"--set {assignment!r}: cannot parse value: {exc}") from exc
    return key, value


def apply_set(cfg: dict[str, Any], key: str, value: Any, origins: dict[str, str]) -> None:
    """Apply a dotted-path assignment in place."""
    parts = key.split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            if nxt is not None:
                raise ConfigError(
                    f"--set {key}=...: '{'.'.join(parts[: i + 1])}' is a "
                    f"{type(nxt).__name__}, not a mapping"
                )
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
    origins[key] = "--set"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


def resolve_machine_path(name_or_path: str, base_dir: Path | None = None) -> Path:
    """A machine may be named (shipped profile) or given as a path.

    A relative path is resolved against the campaign file's own folder, not the
    working directory, so `machine: machine.yaml` means the copy sitting beside
    campaign.yaml however deep in the tree you happen to be standing.
    """
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} or p.is_absolute() or os.sep in name_or_path:
        if not p.is_absolute() and base_dir is not None and not p.is_file():
            candidate = base_dir / p
            if candidate.is_file():
                return candidate
        if not p.is_file():
            where = f" (looked in {base_dir})" if base_dir is not None and not p.is_absolute() else ""
            raise ConfigError(f"machine profile not found: {p}{where}")
        return p
    shipped = MACHINES_DIR / f"{name_or_path}.yaml"
    if not shipped.is_file():
        available = sorted(f.stem for f in MACHINES_DIR.glob("*.yaml"))
        raise ConfigError(
            f"unknown machine {name_or_path!r}; shipped profiles: {available}. "
            f"Pass a path to use your own."
        )
    return shipped


@dataclass
class ResolvedConfig:
    """A fully merged, validated campaign plus the machine it runs on."""

    campaign: Campaign
    machine: Machine
    machine_path: Path
    campaign_path: Path | None
    origins: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        """The campaign folder: everything relative in the config resolves here.

        A campaign is a folder, not a lone file -- machine.yaml, recipe.yaml and
        inputs/ sit beside campaign.yaml -- so a relative path in the config has
        to mean "beside the campaign file", never "beside wherever the user
        happened to type the command".
        """
        return self.campaign_path.parent if self.campaign_path else Path.cwd()

    @property
    def work_dir(self) -> Path:
        """Where this campaign's OUTPUT goes, as an absolute path.

        A relative `workdir` resolves against the campaign folder, so the
        default `workdir: results` puts everything -- database, screen results,
        DFT directories -- inside the campaign folder itself.  ONE place.

        It used to be a required absolute path pointing at a separate scratch
        tree, which left a campaign in two roots plus an `archive` third that
        was configured and never created.  A `results` symlink papered over the
        split, and the split still leaked: the database sat on scratch while
        campaign.yaml and inputs/ sat elsewhere, and `csp status` run from the
        wrong directory reported a different campaign than `csp run`.

        An absolute `workdir` still works and still means exactly what it says,
        for a campaign whose output genuinely belongs on a different filesystem.
        """
        raw = Path(self.campaign.workdir).expanduser()
        return raw if raw.is_absolute() else (self.base_dir / raw).resolve()

    @property
    def campaign_db(self) -> Path:
        """Where this campaign's database lives.

        One property rather than a convention repeated at each call site: the
        driver, the CLI and every stage must agree on it, and a stage that
        guessed differently would quietly build a second, empty campaign.
        """
        return self.work_dir / "campaign.db"

    @property
    def config_hash(self) -> str:
        """Stable hash over the resolved campaign, for provenance.

        Computed from the validated model rather than the raw YAML so that two
        configs which differ only in formatting, key order, or which layer
        supplied a value hash identically -- the hash tracks the physics, not
        the file.
        """
        payload = self.campaign.model_dump(mode="json", exclude_none=False)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.config_hash[:12]


def load_campaign(
    campaign_path: str | Path | None = None,
    *,
    machine: str | None = None,
    template: str | None = None,
    sets: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> ResolvedConfig:
    """Load and validate a campaign through the full layer chain."""
    origins: dict[str, str] = {}
    merged: dict[str, Any] = {}

    # 1. shipped defaults
    if DEFAULTS_PATH.is_file():
        merged = deep_merge(merged, _read_yaml(DEFAULTS_PATH), "defaults.yaml", origins)

    # 2. campaign file, read early only to discover which machine to load
    campaign_data: dict[str, Any] = {}
    cpath: Path | None = None
    if campaign_path is not None:
        cpath = Path(campaign_path)
        if not cpath.is_file():
            raise ConfigError(f"campaign file not found: {cpath}")
        campaign_data = _read_yaml(cpath)

    machine_name = machine or campaign_data.get("machine") or merged.get("machine")
    if not machine_name:
        raise ConfigError(
            "no machine specified: set 'machine:' in the campaign file or pass --machine"
        )
    mpath = resolve_machine_path(str(machine_name), cpath.parent if cpath else None)
    machine_doc = _read_yaml(mpath)

    # A machine profile may carry site-specific campaign defaults under
    # `campaign:`; only that block joins the campaign chain.
    machine_campaign = machine_doc.pop("campaign", {})
    if machine_campaign:
        merged = deep_merge(merged, machine_campaign, f"machine:{mpath.name}", origins)

    # 3. template
    if template:
        tpath = Path(template)
        if not (tpath.suffix in {".yaml", ".yml"} and tpath.is_file()):
            tpath = TEMPLATES_DIR / f"{template}.yaml"
        if not tpath.is_file():
            available = sorted(f.stem for f in TEMPLATES_DIR.glob("*.yaml"))
            raise ConfigError(f"unknown template {template!r}; shipped: {available}")
        merged = deep_merge(merged, _read_yaml(tpath), f"template:{tpath.name}", origins)

    # 4. campaign file
    if campaign_data:
        merged = deep_merge(merged, campaign_data, str(cpath), origins)

    # 5. CLI --set
    for assignment in sets or []:
        key, value = parse_set(assignment)
        apply_set(merged, key, value, origins)

    merged.setdefault("machine", str(machine_name))

    # Expand variables only after every layer has had its say, so a later layer
    # can override a value that would otherwise have needed an undefined var.
    merged = expand_vars(merged, env)
    machine_doc = expand_vars(machine_doc, env)

    try:
        machine_model = Machine.model_validate(machine_doc)
    except Exception as exc:
        raise ConfigError(f"invalid machine profile {mpath}:\n{exc}") from exc

    try:
        campaign_model = Campaign.model_validate(merged)
    except Exception as exc:
        where = str(cpath) if cpath else "<merged config>"
        raise ConfigError(f"invalid campaign config ({where}):\n{exc}") from exc

    return ResolvedConfig(
        campaign=campaign_model,
        machine=machine_model,
        machine_path=mpath,
        campaign_path=cpath,
        origins=origins,
        raw=merged,
    )
