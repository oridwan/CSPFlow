"""Pydantic models for a cspflow campaign -- the single source of truth.

Two rules run through this file and are worth stating once:

1.  ``extra="forbid"`` everywhere.  A mistyped key is an error, not a silently
    ignored line.  The failure this prevents is a campaign that runs to
    completion having quietly ignored the setting you cared about.

2.  Physics values are not invented here.  Where the plan says a value must be
    explicit (``pick``, the INCAR tags, the reference functional), the field is
    required or the default is a deliberate, documented choice recorded in
    provenance -- never a convenience fallback.  See pipeline.md sec.9,
    "What 'fully robust' means here", gate 2.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Rare earths, used by the ``max_rare_earth`` guard (Stage 0.1) and by the
# 4f treatment in Stage 3d.  Sc and Y are excluded: they are group-3 metals with
# no f electrons, so the 4f machinery does not apply to them.
RARE_EARTHS: frozenset[str] = frozenset(
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
)


class Base(BaseModel):
    """Common config: reject unknown keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Stage 0 -- source
# --------------------------------------------------------------------------


class SourceMode(str, Enum):
    chemical_space = "chemical_space"
    composition_list = "composition_list"
    structure_list = "structure_list"


class NStructuresMode(str, Enum):
    fixed = "fixed"
    per_atom = "per_atom"


class NStructures(Base):
    """How many structures to generate for one (composition, Z) row."""

    mode: NStructuresMode = NStructuresMode.per_atom
    count: int | None = Field(None, gt=0, description="mode=fixed: structures per row")
    structures_per_atom: float | None = Field(
        2.0, gt=0, description="mode=per_atom: multiplied by the atom count"
    )

    @model_validator(mode="after")
    def _one_of(self) -> "NStructures":
        if self.mode is NStructuresMode.fixed and self.count is None:
            raise ValueError("n_structures.mode='fixed' requires 'count'")
        if self.mode is NStructuresMode.per_atom and self.structures_per_atom is None:
            raise ValueError("n_structures.mode='per_atom' requires 'structures_per_atom'")
        return self

    def target_for(self, n_atoms: int) -> int:
        if self.mode is NStructuresMode.fixed:
            return int(self.count)  # type: ignore[arg-type]
        return max(1, round(self.structures_per_atom * n_atoms))  # type: ignore[operator]


class ZRange(Base):
    """Formula units of the *reduced* formula to enumerate."""

    min: int = Field(1, ge=1)
    max: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "ZRange":
        if self.max < self.min:
            raise ValueError(f"z.max ({self.max}) < z.min ({self.min})")
        return self

    def values(self) -> list[int]:
        return list(range(self.min, self.max + 1))


class SourceDefaults(Base):
    """Inherited by modes 1 and 2; per-item overrides win."""

    z: ZRange = Field(default_factory=ZRange)
    max_atoms: int = Field(40, gt=0, description="hard cap on Z * atoms per formula unit")
    n_structures: NStructures = Field(default_factory=NStructures)
    n_structures_scope: Literal["per_z", "total"] = "per_z"


class ElementGroup(Base):
    """One group in a chemical space, e.g. ``{Fe, Co, Ni}`` at >= 0.75 fraction.

    ``pick`` is REQUIRED and has no default on purpose.  ``{Fe,Co,Ni}-{Y,Gd}``
    is ambiguous: one element per group gives 6 binary systems, allowing two
    from a group adds ternaries and a 5-10x larger sweep.  Left implicit this is
    a silent campaign-size multiplier, so the user states it.  See pipeline.md
    sec.0.1.
    """

    elements: list[str] = Field(..., min_length=1)
    pick: int | list[int] = Field(
        ..., description="how many elements to take from this group; int or list of arities"
    )
    min_fraction: float | None = Field(None, ge=0.0, le=1.0)
    max_fraction: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("elements")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            dupes = sorted({e for e in v if v.count(e) > 1})
            raise ValueError(f"duplicate elements in group: {dupes}")
        return v

    @field_validator("pick")
    @classmethod
    def _positive(cls, v: int | list[int]) -> int | list[int]:
        arities = [v] if isinstance(v, int) else v
        if not arities:
            raise ValueError("pick must not be empty")
        for a in arities:
            if a < 1:
                raise ValueError(f"pick must be >= 1, got {a}")
        return v

    @model_validator(mode="after")
    def _checks(self) -> "ElementGroup":
        if (
            self.min_fraction is not None
            and self.max_fraction is not None
            and self.min_fraction > self.max_fraction
        ):
            raise ValueError(
                f"min_fraction ({self.min_fraction}) > max_fraction ({self.max_fraction})"
            )
        for a in self.arities():
            if a > len(self.elements):
                raise ValueError(
                    f"pick={a} exceeds the {len(self.elements)} elements in the group"
                )
        return self

    def arities(self) -> list[int]:
        return [self.pick] if isinstance(self.pick, int) else list(self.pick)


class ChemicalSpace(Base):
    """Mode 1: element groups plus per-group ratio constraints."""

    groups: dict[str, ElementGroup] = Field(..., min_length=1)
    max_atoms_formula: int = Field(20, gt=0, description="cap on the reduced formula itself")
    max_rare_earth: int | None = Field(
        1,
        ge=0,
        description=(
            "max rare-earth species in the ASSEMBLED system.  Deliberately separate "
            "from `pick`, which is arity within one group: nothing forces the rare "
            "earths into their own group, so a user writing {Sm, Tb, Fe} with "
            "pick=[1,2] would otherwise generate Sm-Tb-Fe unasked.  null = no limit."
        ),
    )


class CompositionItem(Base):
    """One explicit formula in mode 2, with optional per-item overrides."""

    formula: str = Field(..., min_length=1)
    z: ZRange | None = None
    max_atoms: int | None = Field(None, gt=0)
    n_structures: NStructures | None = None

    @field_validator("z", mode="before")
    @classmethod
    def _z_shorthand(cls, v: Any) -> Any:
        """Accept ``z: [1, 2]`` as shorthand for ``z: {min: 1, max: 2}``."""
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"z as a list must be [min, max], got {list(v)}")
            return {"min": v[0], "max": v[1]}
        return v


class CompositionList(Base):
    """Mode 2: explicit formulas, still generated."""

    items: list[CompositionItem] = Field(default_factory=list)
    from_file: str | None = Field(
        None, description="CSV: formula[,z_min,z_max,n_structures]"
    )

    @model_validator(mode="after")
    def _something(self) -> "CompositionList":
        if not self.items and self.from_file is None:
            raise ValueError("composition_list needs either 'items' or 'from_file'")
        return self


class StructureList(Base):
    """Mode 3: structures in, NO generation.

    ``dedup`` defaults to ``warn`` deliberately.  For generated structures,
    dropping duplicates is the entire point of Stage 2.  For a curated input
    list it is a hazard: two seeds you supplied on purpose -- a relaxed and an
    unrelaxed copy of one prototype, say -- would be silently merged and you
    would never learn which survived.  See pipeline.md sec.0.3.

    ``off`` is the third option, and it exists because the comparison is not
    free.  `StructureMatcher.fit` costs ~320 ms on a 9-atom five-species cell,
    and the pairs go as k-squared WITHIN each formula group: 232 CePdGe seeds
    with four 14-member groups came to 739 fits and **237 seconds**, against
    6.8 s to parse and validate all 232 files.  Use ``off`` when the overlap is
    deliberate and already recorded elsewhere -- a campaign whose seeds come
    from two design routes on purpose, with a provenance CSV saying which is
    which, learns nothing from being told they overlap.
    """

    paths: list[str] = Field(..., min_length=1, description="POSCAR/CIF paths or globs")
    relax: bool = Field(True, description="MLIP-relax the seed before DFT")
    dedup: Literal["warn", "drop", "off"] = "warn"
    max_atoms: int | None = Field(None, gt=0, description="defaults to source.defaults.max_atoms")

    @field_validator("dedup", mode="before")
    @classmethod
    def _dedup_off_is_a_word(cls, v: Any) -> Any:
        """`dedup: off`, written as documented, reaches here as the boolean False:
        YAML 1.1 reads a bare `off` that way. Same trap and same fix as
        `on_fail` (D065) -- translate it, because the user is not wrong."""
        return "off" if v is False else v


class Source(Base):
    """One entry in the ``source:`` list.

    ``name`` is required when several sources are present: without it, "did this
    candidate come from the sweep or from my seed list?" is unanswerable once the
    rows are pooled, and the control group stops being a control group.
    """

    mode: SourceMode
    name: str = Field("default", min_length=1)
    defaults: SourceDefaults = Field(default_factory=SourceDefaults)
    chemical_space: ChemicalSpace | None = None
    composition_list: CompositionList | None = None
    structure_list: StructureList | None = None

    @model_validator(mode="after")
    def _block_matches_mode(self) -> "Source":
        blocks = {
            SourceMode.chemical_space: self.chemical_space,
            SourceMode.composition_list: self.composition_list,
            SourceMode.structure_list: self.structure_list,
        }
        if blocks[self.mode] is None:
            raise ValueError(f"source.mode='{self.mode.value}' requires a '{self.mode.value}:' block")
        extra = [m.value for m, b in blocks.items() if b is not None and m is not self.mode]
        if extra:
            raise ValueError(
                f"source.mode='{self.mode.value}' but these blocks are also set: {extra}. "
                "Use a separate list entry for each source rather than one entry with several blocks."
            )
        return self

    @property
    def entry_stage(self) -> Literal["generate", "screen"]:
        """Where this source's rows enter the funnel.

        Mode 3 is an entry point, not a branch: it emits structures directly and
        skips generation, but everything from Stage 2 on is composition-agnostic
        and needs no change.
        """
        return "screen" if self.mode is SourceMode.structure_list else "generate"


# --------------------------------------------------------------------------
# Stages 1-2 -- generate, screen
# --------------------------------------------------------------------------


class Resources(Base):
    """Scheduler ask for one stage.  Resolved against the machine profile."""

    role: str = Field("cpu", description="machine-profile partition role: cpu | gpu | ...")
    ntasks: int | None = Field(None, gt=0)
    cpus_per_task: int | None = Field(None, gt=0)
    gpus: int | None = Field(None, ge=0)
    mem: str | None = None
    time: str = "24:00:00"

    @field_validator("time")
    @classmethod
    def _walltime(cls, v: str) -> str:
        # SLURM accepts several forms; we require D-HH:MM:SS or HH:MM:SS so that
        # the driver can compare walltimes against QOS limits numerically.
        import re

        if not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", v):
            raise ValueError(f"time must be [D-]HH:MM:SS, got {v!r}")
        return v


class MatterGen(Base):
    model: str = Field(..., description="checkpoint directory")
    mode: Literal["csp", "unconditional"] = "csp"
    max_batch_size: int = Field(100, gt=0)
    timeout_per_batch: int = Field(1800, gt=0, description="seconds")


class Generate(Base):
    engine: Literal["mattergen"] = "mattergen"
    mattergen: MatterGen | None = None
    resources: Resources = Field(default_factory=lambda: Resources(role="gpu", gpus=1))

    @model_validator(mode="after")
    def _engine_block(self) -> "Generate":
        if self.engine == "mattergen" and self.mattergen is None:
            raise ValueError("generate.engine='mattergen' requires a 'mattergen:' block")
        return self


class MatterSim(Base):
    model: str = "MatterSim-v1.0.0-5M.pth"
    fmax: float = Field(0.01, gt=0, description="eV/A force convergence")
    max_steps: int = Field(500, gt=0)

    # NOTE: there is no `batch_size`. Relaxation is ONE STRUCTURE AT A TIME.
    #
    # It was here, defaulted to 32, and read by nothing -- so it described GPU
    # batching that does not happen. MatterSim's own `BatchRelaxer` is unusable
    # against the installed ASE (two independent breakages, one of them inside
    # mattersim's own loop), and it accepts no step limit, so an unconvergeable
    # structure would have nothing to stop it. `mlip/mattersim_engine.py` runs
    # ASE's optimizer against `MatterSimCalculator` instead, which honours
    # `max_steps` and lets us own the convergence test. See that module's
    # docstring for the measurements.
    #
    # A knob wired to nothing is worse than no knob: it spends the reader's
    # attention and implies a capability that is not there.

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_batch_size(cls, data):
        """Accept and ignore `batch_size` from an older campaign.yaml.

        `extra="forbid"` would otherwise make a file that sets it fail to load
        entirely. Refusing to open a campaign over a setting that never did
        anything is a worse outcome than opening it and saying so.
        """
        if isinstance(data, dict) and "batch_size" in data:
            data = {k: v for k, v in data.items() if k != "batch_size"}
            warnings.warn(
                "screen.mattersim.batch_size is retired and was ignored: MLIP "
                "relaxation runs one structure at a time, so it never had an "
                "effect. Remove it from campaign.yaml.",
                UserWarning, stacklevel=2,
            )
        return data


class StructureMatcherCfg(Base):
    ltol: float = Field(0.2, gt=0)
    stol: float = Field(0.2, gt=0)
    angle_tol: float = Field(5.0, gt=0)


class Dedup(Base):
    matcher: StructureMatcherCfg = Field(default_factory=StructureMatcherCfg)


class CHGNet(Base):
    """CHGNet's screen settings.

    `fmax` defaults looser than MatterSim's 0.01. CHGNet is carried for its
    MOMENTS, not to be the final geometry, and the moment is far less sensitive
    to the last few meV/A than the energy is. Tighten it deliberately if CHGNet
    is doing the relaxation rather than reading one.
    """

    model: str = Field("", description="path to weights; empty = CHGNet's shipped model")
    fmax: float = Field(0.05, gt=0, description="eV/A force convergence")
    max_steps: int = Field(500, gt=0)
    mask_elements: list[str] = Field(
        default_factory=list,
        description="elements whose predicted moment is DISCARDED when summing. "
                    "CHGNet inherits MP's f-in-valence convention, so its Ce/Gd "
                    "moments are not what a frozen-f campaign reports [D1]. "
                    "Masking is a deliberate choice and is recorded per seed.",
    )


class Screen(Base):
    mlip: Literal["mattersim", "mace", "uma", "chgnet"] = "mattersim"
    mattersim: MatterSim = Field(default_factory=MatterSim)
    chgnet: CHGNet = Field(default_factory=CHGNet)
    dedup: Dedup = Field(default_factory=Dedup)
    resources: Resources = Field(default_factory=lambda: Resources(role="gpu", gpus=1))


# --------------------------------------------------------------------------
# Stage 3 -- reference
# --------------------------------------------------------------------------


class ThermoType(str, Enum):
    """MP's own functional label.

    This must be pinned.  MP's /materials/summary/ endpoint returns
    ``uncorrected_energy_per_atom`` from whichever functional it prefers per
    material, with NO field in the response saying which -- SmFe2 comes back as
    -7.1966 (GGA) or -19.4095 (r2SCAN) depending on the material.  Mixing them
    inside one chemsys puts a 5-12 eV/atom discontinuity into the hull.  See
    pipeline.md sec.3.1.
    """

    GGA_GGA_U = "GGA_GGA+U"
    R2SCAN = "R2SCAN"
    GGA_GGA_U_R2SCAN = "GGA_GGA+U_R2SCAN"


class Reference(Base):
    functionals: list[Literal["GGA", "GGA+U"]] = Field(default_factory=lambda: ["GGA"])
    thermo_type: ThermoType = Field(
        ThermoType.GGA_GGA_U,
        description="PINNED. A reference set containing more than one is refused.",
    )
    energy_scale: Literal["raw", "mp_corrected"] = "raw"
    mode: Literal["mp_energies", "recompute"] = "recompute"
    energy_source: Literal["dft", "mlip", "mp"] = Field(
        "dft",
        description=(
            "Which of the store's three energies builds the hull, when "
            "`mode: recompute`. Every structure in $CSPFLOW_STORE carries all "
            "three and each defines a hull on its own scale:\n"
            "  dft   our VASP recompute   (the default; what candidates are ranked on)\n"
            "  mlip  MatterSim as we ran it\n"
            "  mp    Materials Project's own numbers\n"
            "They are NEVER mixed. `dft` and `mlip` are ours -- same settings, "
            "same store. `mp` is a different scale (ours minus MP is +0.15 to "
            "+0.21 eV/atom against a 0.06 eV/atom selection threshold), so it "
            "is for comparing against, not for filling a gap in another hull."
        ),
    )
    prescreen_mode: Literal["mp_energies", "recompute"] = "mp_energies"
    prescreen_hull_max: float = Field(0.20, gt=0, description="eV/atom, widened for Phase A")
    snapshot: bool = Field(
        True,
        description=(
            "Freeze the MP query.  Without it, e_above_hull moves with no change "
            "to any of our inputs, which is indistinguishable from a bug."
        ),
    )
    snapshot_id: str = Field("auto", description="'auto' = new, stamped with date + MP release")
    cache: str = "$CSPFLOW_REFERENCE/mp"
    recompute_cache: str = "$CSPFLOW_REFERENCE/computed"
    relax_with_mlip: bool = True


# --------------------------------------------------------------------------
# Stage 4 -- calibrate
# --------------------------------------------------------------------------


class OnFail(str, Enum):
    off = "off"
    warn = "warn"
    block = "block"


def _on_fail(value: Any) -> Any:
    """Undo YAML 1.1's boolean coercion of `off`.

    PyYAML implements YAML 1.1, where bare `off`, `no` and `false` are all the
    boolean False -- so `on_fail: off`, written exactly as the documentation
    says, reaches pydantic as `False` and is rejected with a message about an
    enum that mentions neither YAML nor booleans.

    The same trap bit the DFT retry ladder, where `on:` became `True:` and a
    retry rule silently had no condition (D065). There it was caught by
    refusing the key; here the value is simply translated, because `off` is a
    legitimate thing to write and the user is not wrong.
    """
    if value is False:
        return "off"
    if value is True:
        raise ValueError(
            "on_fail must be 'off', 'warn' or 'block'. YAML 1.1 reads a bare "
            "`on` as the boolean True; quote it if you meant a word.")
    return value


class CalibrateMPThresholds(Base):
    mae_e_per_atom: float = Field(0.05, gt=0, description="single-point MLIP on MP geometry")
    spearman_min: float = Field(0.90, ge=-1.0, le=1.0)
    max_volume_drift: float = Field(
        0.05, gt=0, description="geometry error, kept as its own number, never folded into the MAE"
    )


class CalibratePilotThresholds(Base):
    mae_e_per_atom: float = Field(0.05, gt=0)
    mae_e_hull: float = Field(0.05, gt=0)
    spearman_min: float = Field(0.90, ge=-1.0, le=1.0)


class CalibrateMP(Base):
    """4a: free.  MP's own DFT is the yardstick, at FIXED geometry.

    Single-point, never relaxed-vs-relaxed: comparing E_MLIP(x_MLIP) against
    E_DFT(x_MP) folds energy error and geometry error into one inseparable MAE,
    and the two have opposite consequences.  A uniform energy offset largely
    cancels in a hull; a volume bias does not and is fatal for screening.
    """

    on_fail: OnFail = OnFail.warn
    thresholds: CalibrateMPThresholds = Field(default_factory=CalibrateMPThresholds)

    _fix_on_fail = field_validator("on_fail", mode="before")(_on_fail)


class CalibratePilot(Base):
    """4b: costs pilot DFT.  OUR DFT is the yardstick.  OFF by default (D119).

    It was the gate, and it was the right gate when the only calibration
    available was 4a -- MatterSim against *MP's* numbers, which for an
    MPtrj-trained model is largely a self-consistency check.

    That is no longer the only calibration.  The reference set now carries both
    MatterSim and our own DFT for 2,711 phases, so the comparison 4b existed to
    make -- MLIP against OUR DFT -- is already available before a campaign
    starts, at 68x the sample size 4b's default 40 would have given:

        MatterSim vs our DFT, formation energy   MAE 0.0355 eV/atom
                                                 median |err| 0.0148
                                                 Spearman 0.970

    So the check happens earlier and on more data, and the pilot's DFT is spent
    on candidates instead.

    **What is genuinely given up, and it is not nothing.**  Those 2,711 phases
    are MP phases, which are near in-distribution for this class of model.  The
    generated structures are the out-of-distribution set, and 4b was the only
    thing that tested the model there.  Turning it on remains the honest move
    for a campaign in a chemistry the reference set does not cover, or one whose
    generator is asked for prototypes unlike anything in MP:

        calibrate: {pilot: {on_fail: block}}
    """

    on_fail: OnFail = OnFail.off
    pilot_n: int = Field(40, gt=0, description="screened candidates given pilot DFT")
    thresholds: CalibratePilotThresholds = Field(default_factory=CalibratePilotThresholds)

    _fix_on_fail = field_validator("on_fail", mode="before")(_on_fail)


class Calibrate(Base):
    mp: CalibrateMP = Field(default_factory=CalibrateMP)
    pilot: CalibratePilot = Field(default_factory=CalibratePilot)


# --------------------------------------------------------------------------
# Stage 5 -- filter
# --------------------------------------------------------------------------


class SpacegroupFilter(Base):
    min_number: int = Field(1, ge=1, le=230)


class Filter(Base):
    e_above_hull_max: float = Field(0.10, gt=0, description="eV/atom")
    e_above_hull_max_source: Literal["literal", "calibrated"] = Field(
        "calibrated",
        description=(
            "'calibrated' rescales the THRESHOLD by the fitted alpha/beta from "
            "Stage 4a rather than rewriting any stored energy.  If the MLIP "
            "compresses hull distances by 1.4x, screening at 0.10 in MLIP units "
            "silently discards candidates sitting at 0.10 in DFT units."
        ),
    )
    max_per_composition: int = Field(5, gt=0)
    spacegroup: SpacegroupFilter = Field(default_factory=SpacegroupFilter)


# --------------------------------------------------------------------------
# Stage 6 -- dft
# --------------------------------------------------------------------------


class PotcarCfg(Base):
    """Both POTCAR trees stay available; the choice is the user's per campaign.

    Whichever is chosen, every resolved POTCAR is pinned by md5_header_hash in
    provenance, so the two trees can never be mixed inside one hull -- that is a
    hard refusal, not a warning.  See pipeline.md sec.3c.5.
    """

    tree: Literal["VASP6.4", "VASP5.2"] = "VASP6.4"
    functional: str = Field("PBE_64", description="pymatgen functional label")
    overrides: dict[str, str] = Field(
        default_factory=dict, description="element -> POTCAR symbol, e.g. {Gd: Gd_3}"
    )


class FTreatment(str, Enum):
    """Frozen 4f (RE_3 across the whole series) or f in valence.

    Both are supported and switchable per campaign.  ``frozen`` ships as the
    default for the screening funnel: it is faster, converges, and introduces no
    free U parameter.  ``valence`` is what you need for real moments and any
    future MAE work.  The one thing that is forbidden is mixing them within one
    campaign -- which is exactly what MPRelaxSet does across the RE series.
    """

    frozen = "frozen"
    valence = "valence"


class RareEarth(Base):
    f_treatment: FTreatment = FTreatment.frozen
    magnetic_order: Literal["ferri", "ferro", "none"] = "ferri"
    reconstruct_ms: bool = Field(
        True, description="add Hund's-rule 4f moments back in Stage 7, reported separately"
    )


class Magnetism(Base):
    mode: Literal[
        "none", "pymatgen", "table", "ferrimagnetic_retm", "ferromagnetic"
    ] = "ferrimagnetic_retm"
    strict: bool = Field(
        True, description="fail if any site would take a default MAGMOM"
    )
    table: dict[str, float] = Field(default_factory=dict, description="element -> initial moment")
    site_overrides: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _materialise(self) -> "Magnetism":
        """Expand a named mode into the literal table it stands for.

        `recipe_id` hashes this model.  A mode NAME in the hash would record
        that the moments came from `FERRO_RETM` but not what `FERRO_RETM` said,
        so editing that constant would change every energy while leaving the
        cache key -- and therefore the claim that two energies are comparable --
        untouched.  Materialising here puts the numbers themselves in the hash:
        change a moment, get a different recipe_id, miss the cache correctly.

        Anything the user wrote in `table` wins, so an override stays an
        override rather than being overwritten by the defaults it overrides.
        """
        if self.mode == "ferromagnetic":
            from ..dft.vasp.incar import FERRO_RETM

            merged = {**FERRO_RETM, **self.table}
            object.__setattr__(self, "table", dict(sorted(merged.items())))
        return self


class Ldau(Base):
    enabled: bool = False
    u: dict[str, float] = Field(default_factory=dict, description="element -> U (eV)")
    j: dict[str, float] = Field(default_factory=dict)
    ldau_type: int = Field(2, ge=1, le=4)


class Select(Base):
    """Which screened candidates get DFT, and in what order.

    `budget_core_hours` used to live here and is retired (D143). A core-hour
    budget gates on a PROJECTION -- a per-stage mean cost multiplied by what is
    queued -- and before any job of a stage has finished that projection is the
    requested walltime, which is the ceiling rather than the cost. On the
    RE-magnets campaign that made a 20,000 core-hour budget halt the run after
    eight of 150 structures, while nothing was actually overspent. What a run
    needs protected is the cores it holds right now, which is `dft.max_cores`
    (D142): a measured quantity, not an extrapolation from one.
    """

    rank_by: str = "e_above_hull_mlip"
    max_per_composition: int = Field(3, gt=0)
    max_total: int = Field(1500, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_budget(cls, data):
        """Accept and ignore `budget_core_hours` from an older campaign.yaml.

        `extra="forbid"` would make every campaign that sets it -- which is
        every campaign written before today -- fail to load outright. Opening
        the file and saying so is the better outcome.
        """
        if isinstance(data, dict) and "budget_core_hours" in data:
            data = {k: v for k, v in data.items() if k != "budget_core_hours"}
            warnings.warn(
                "dft.select.budget_core_hours is retired (D143) and was "
                "ignored. It gated on a projected cost, which before any job "
                "finished was the requested walltime, so it halted runs that "
                "had not overspent. Cap cores instead: dft.max_cores. "
                "Remove it from campaign.yaml.",
                UserWarning, stacklevel=2,
            )
        return data


class Dft(Base):
    recipe: str = Field("magnets", description="name in dft/recipes/, or a path to a YAML file")
    potcar: PotcarCfg = Field(default_factory=PotcarCfg)
    rare_earth: RareEarth = Field(default_factory=RareEarth)
    magnetism: Magnetism = Field(default_factory=Magnetism)
    ldau: Ldau = Field(default_factory=Ldau)
    nbands: Literal["auto"] | int = "auto"
    incar_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Campaign-level INCAR overrides applied on top of the recipe, for every "
            "stage.  Free-form: there is no whitelist.  Unknown tags are written as "
            "given; unrecognised ones warn but are never silently dropped."
        ),
    )
    max_in_flight: int = Field(200, gt=0, description="jobs submitted at once (QOS-aware)")
    max_concurrent_tasks: int = Field(
        48, gt=0,
        description=(
            "How many of this campaign's DFT jobs may RUN at once. It was the "
            "`--array %N` throttle; with one job per structure (D135) there is "
            "no array to throttle, so the site's QOS enforces it directly "
            "(MaxTRESPU cpu=768 / ntasks=16 = 48 on orion). It still bounds "
            "what the driver reports and, on a scheduler that models arrays, "
            "what any remaining array submission is capped at. It can never "
            "exceed max_in_flight: the throttle is min(concurrent, in_flight)."
        ),
    )
    max_cores: int | None = Field(
        None, gt=0,
        description=(
            "Total cores this campaign's DFT jobs may hold at once, counting "
            "QUEUED as well as RUNNING. The cap people actually want: "
            "`max_in_flight` counts JOBS, so at 64 ranks a limit of 48 is 3,072 "
            "cores -- and nothing on this cluster's cpu role caps that. Queued "
            "jobs count because they will occupy those cores; a cap that ignored "
            "them would approve a submission the queue has already spent. "
            "Only this campaign's own jobs are counted, never anyone else's. "
            "null = no core cap (job counts alone decide)."
        ),
    )
    select: Select = Field(default_factory=Select)
    layout: Literal["runs", "stages"] = Field(
        # 'runs' IS THE DEFAULT (changed 2026-09-14, D135). It was 'stages'
        # only because CeFeB and CePdGe were mid-flight without a `layout:`
        # key, and re-pointing a running campaign orphans its finished work.
        # Those campaigns are retired, so the default now matches the design
        # rather than the transition.
        #
        # An old campaign is still safe: `layout.resolve()` lets the directories
        # on disk override this, so a tree of `dft-<id>-<step>/` keeps 'stages'
        # and says why. Configuration is an intention; the directories are a
        # fact.
        "runs",
        description=(
            "How DFT work is arranged on disk. 'runs' (the default for a new "
            "campaign) gives each STRUCTURE one directory holding its seed, its "
            "script, its log and one subdirectory per step. 'stages' is the older "
            "flat arrangement -- one directory per structure per step, with the "
            "scripts, manifests, claims and logs beside them -- which put 428 "
            "entries at one level for a 94-structure campaign. Existing campaigns "
            "keep 'stages'; it is not changed under a running driver."
        ),
    )
    combined_job: bool = Field(
        True,
        description=(
            "Run a structure's steps in ONE job instead of one job per step. "
            "Removes the queue wait between them -- measured at 54 min median and "
            "94 min mean over 29 CeFeB pairs, 45.7 hours of idle across those "
            "structures alone -- and makes every array homogeneous, so a batch "
            "can no longer mix a 24-hour relax with a 12-hour static and give "
            "both the shorter walltime. Requires layout='runs'."
        ),
    )


# --------------------------------------------------------------------------
# Stage 7 -- analyze
# --------------------------------------------------------------------------


class Analyze(Base):
    properties: list[str] = Field(
        default_factory=lambda: ["m_dft_raw", "m_s_reconstructed", "volume", "spacegroup"]
    )
    report: Literal["html", "none"] = "html"

    @model_validator(mode="after")
    def _never_merge_moments(self) -> "Analyze":
        """m_dft_raw and m_s_reconstructed are reported side by side, never merged.

        The reconstructed value is a model layered on DFT, not a computed
        result.  Collapsing them into one column is how a Hund's-rule estimate
        ends up quoted as a DFT number.
        """
        props = set(self.properties)
        if "m_s" in props:
            raise ValueError(
                "'m_s' is ambiguous: use 'm_dft_raw' (computed) and/or "
                "'m_s_reconstructed' (Hund's-rule model). They are never merged."
            )
        return self


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


class Campaign(Base):
    """A whole campaign.  This is the only file a user edits."""

    name: str = Field(..., min_length=1)
    machine: str = Field(..., min_length=1, description="machine profile name or path")
    workdir: str = Field(..., min_length=1)
    archive: str | None = Field(
        None,
        description=(
            "db + report mirrored here at every stage boundary, so a scratch purge "
            "costs compute but never provenance"
        ),
    )

    source: list[Source] = Field(..., min_length=1)
    generate: Generate | None = None
    screen: Screen = Field(default_factory=Screen)
    reference: Reference = Field(default_factory=Reference)
    calibrate: Calibrate = Field(default_factory=Calibrate)
    filter: Filter = Field(default_factory=Filter)
    dft: Dft = Field(default_factory=Dft)
    analyze: Analyze = Field(default_factory=Analyze)

    @field_validator("source", mode="before")
    @classmethod
    def _accept_bare_mapping(cls, v: Any) -> Any:
        """A single source may be written as a bare mapping instead of a 1-list."""
        if isinstance(v, dict):
            return [v]
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> "Campaign":
        names = [s.name for s in self.source]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"duplicate source names {dupes}. Every source needs a distinct 'name:' -- "
                "it is stamped on each row it emits, and without it a pooled campaign "
                "cannot say which source a candidate came from."
            )
        if len(self.source) > 1 and any(s.name == "default" for s in self.source):
            raise ValueError(
                "with more than one source, every entry needs an explicit 'name:' "
                "(one is still using the placeholder 'default')"
            )
        if self.needs_generation and self.generate is None:
            modes = sorted({s.mode.value for s in self.source if s.entry_stage == "generate"})
            raise ValueError(
                f"source mode(s) {modes} generate structures, so a 'generate:' block is required"
            )
        return self

    @property
    def needs_generation(self) -> bool:
        return any(s.entry_stage == "generate" for s in self.source)

    @property
    def rare_earth_elements(self) -> set[str]:
        """Rare earths mentioned anywhere in the campaign's chemical spaces."""
        found: set[str] = set()
        for s in self.source:
            if s.chemical_space:
                for g in s.chemical_space.groups.values():
                    found |= RARE_EARTHS & set(g.elements)
        return found


# --------------------------------------------------------------------------
# Machine profile -- the portability answer (pipeline.md sec.6.5)
# --------------------------------------------------------------------------


class Partition(Base):
    name: str = Field(..., description="site partition name(s), comma-separated for SLURM")
    qos: str | None = None
    account: str | None = None
    constraint: str | None = None
    exclude: str | None = None


class PartitionLimits(Base):
    max_submit: int | None = Field(None, gt=0)
    max_cpus: int | None = Field(None, gt=0)
    max_gpus: int | None = Field(None, gt=0)
    max_walltime: str | None = None


class MachineDefaults(Base):
    nodes: int = Field(1, gt=0)
    ntasks: int = Field(16, gt=0)
    cpus_per_task: int = Field(1, gt=0)
    mem: str = "32G"


class Codes(Base):
    vasp_std: str | None = None
    vasp_gam: str | None = None
    vasp_ncl: str | None = None
    mpi_launcher: str = "srun --mpi=pmi2"


class Machine(Base):
    """A site profile.  Everything site-specific lives here and nowhere else."""

    scheduler: Literal["slurm", "local"] = "slurm"
    partitions: dict[str, Partition] = Field(default_factory=dict)
    defaults: MachineDefaults = Field(default_factory=MachineDefaults)
    limits: dict[str, PartitionLimits] = Field(default_factory=dict)
    modules: dict[str, list[str]] = Field(default_factory=dict)
    env: dict[str, str | int] = Field(default_factory=dict)
    codes: Codes = Field(default_factory=Codes)
    potcar_root: str | None = Field(
        None, description="pymatgen-layout tree (PMG_VASP_PSP_DIR)"
    )
    potcar_dirs: dict[str, str] = Field(
        default_factory=dict, description="functional label -> pymatgen directory name"
    )
    potcar_trees: dict[str, str] = Field(
        default_factory=dict, description="tree label (VASP6.4/VASP5.2) -> functional label"
    )
    conda: dict[str, str] = Field(default_factory=dict, description="role -> env name")
    scratch: str = "/scratch/$USER"

    def partition_for(self, role: str) -> Partition:
        if role not in self.partitions:
            known = sorted(self.partitions) or ["<none defined>"]
            raise KeyError(
                f"machine profile has no partition for role {role!r}; known roles: {known}"
            )
        return self.partitions[role]
