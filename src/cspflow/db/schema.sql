-- cspflow campaign state.
--
-- This schema lives in the SAME SQLite file as the ASE database.  An ase.db
-- .db file IS a SQLite file (tables: systems, keys, species, text_key_values,
-- number_key_values, information, sqlite_sequence), and adding our own tables
-- alongside is safe -- ASE reads and writes normally afterwards.  That gets us
-- both halves of the state model: ASE owns structures, so `ase gui` and PyXtal
-- read a live campaign with no export step, while the relational tables below
-- own campaign state, which ASE's flat row store models badly.
--
-- Two conventions, forced on us by measured ASE behaviour (see pipeline.md
-- sec.7):
--
--   * Pending work is found by an explicit `state` string, NEVER by a missing
--     key.  ASE's db.select('~key') does not return the complement, so
--     "not yet computed" as an absent key is a dead end -- the failure mode is
--     a driver loop that silently never picks up any work.
--   * No NULL for "not measured yet".  A failed or skipped measurement carries
--     a state plus a reason.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS campaign_meta (
    key            TEXT PRIMARY KEY,
    value          TEXT NOT NULL
);

-- Compositions have no Atoms object, so they cannot be ASE rows.
CREATE TABLE IF NOT EXISTS composition (
    id             INTEGER PRIMARY KEY,
    formula        TEXT    NOT NULL,          -- reduced formula
    chemsys        TEXT    NOT NULL,          -- '-'-joined sorted elements
    z              INTEGER NOT NULL,          -- formula units
    n_atoms        INTEGER NOT NULL,          -- z * atoms per formula unit
    n_target       INTEGER NOT NULL DEFAULT 0,-- structures wanted
    n_produced     INTEGER NOT NULL DEFAULT 0,-- structures the generator returned
    source_mode    TEXT    NOT NULL,
    source_name    TEXT    NOT NULL,
    state          TEXT    NOT NULL DEFAULT 'new'
                   CHECK (state IN ('new','generating','generated','done','failed','skipped')),
    fail_reason    TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (formula, z, source_name)
);
CREATE INDEX IF NOT EXISTS ix_composition_state   ON composition(state);
CREATE INDEX IF NOT EXISTS ix_composition_chemsys ON composition(chemsys);

-- One MLIP or VASP relaxation of one structure.
CREATE TABLE IF NOT EXISTS relaxation (
    id             INTEGER PRIMARY KEY,
    structure_id   INTEGER NOT NULL,          -- -> ASE systems.id
    engine         TEXT    NOT NULL,          -- 'mattersim' | 'vasp:relax' | ...
    energy         REAL,
    e_per_atom     REAL,
    converged      INTEGER NOT NULL DEFAULT 0,
    n_steps        INTEGER NOT NULL DEFAULT 0,
    volume_before  REAL,
    volume_after   REAL,
    provenance_id  INTEGER REFERENCES provenance(id),
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_relaxation_structure ON relaxation(structure_id);

-- MP reference phases.  thermo_type is recorded per entry and a reference set
-- containing more than one is refused: MP's summary endpoint silently mixes
-- functionals, and SmFe2 differs by 12.2 eV/atom between GGA and r2SCAN.
CREATE TABLE IF NOT EXISTS reference_entry (
    id                INTEGER PRIMARY KEY,
    mp_id             TEXT    NOT NULL,
    chemsys           TEXT    NOT NULL,
    thermo_type       TEXT    NOT NULL,
    run_type          TEXT    NOT NULL DEFAULT '',
    formula           TEXT    NOT NULL DEFAULT '',
    n_atoms           INTEGER NOT NULL DEFAULT 0,
    e_dft_raw         REAL,
    e_dft_corrected   REAL,
    correction        REAL,
    e_mlip_static     REAL,                   -- single-point at MP geometry
    e_mlip_relaxed    REAL,                   -- after MLIP relaxation
    volume_drift      REAL,                   -- geometry error, kept separate
    rmsd              REAL,
    structure_id      INTEGER,
    snapshot_id       TEXT    NOT NULL DEFAULT '',
    state             TEXT    NOT NULL DEFAULT 'fetched'
                      CHECK (state IN ('fetched','static_done','relaxed','failed')),
    fail_reason       TEXT    NOT NULL DEFAULT '',
    UNIQUE (mp_id, thermo_type, snapshot_id)
);
CREATE INDEX IF NOT EXISTS ix_reference_chemsys ON reference_entry(chemsys);

-- Hull placement.  hull_type and energy_scale are part of the identity: two
-- energies computed on different scales may never enter the same hull.
CREATE TABLE IF NOT EXISTS hull (
    id               INTEGER PRIMARY KEY,
    structure_id     INTEGER NOT NULL,
    hull_type        TEXT    NOT NULL CHECK (hull_type IN ('mlip','dft')),
    energy_scale     TEXT    NOT NULL CHECK (energy_scale IN ('raw','mp_corrected')),
    e_above_hull     REAL    NOT NULL,
    formation_energy REAL,
    ref_set_hash     TEXT    NOT NULL,
    settings_hash    TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (structure_id, hull_type, energy_scale, ref_set_hash)
);
CREATE INDEX IF NOT EXISTS ix_hull_structure ON hull(structure_id);
CREATE INDEX IF NOT EXISTS ix_hull_ranking   ON hull(hull_type, energy_scale, e_above_hull);

-- One unit of scheduled work: one structure at one recipe stage.
CREATE TABLE IF NOT EXISTS job (
    id             INTEGER PRIMARY KEY,
    structure_id   INTEGER,
    stage          TEXT    NOT NULL,
    recipe_step    TEXT    NOT NULL DEFAULT '',
    slurm_id       TEXT    NOT NULL DEFAULT '',
    array_task_id  INTEGER,
    state          TEXT    NOT NULL DEFAULT 'pending'
                   CHECK (state IN ('pending','queued','running','done','failed',
                                    'cancelled','timeout','held')),
    attempt        INTEGER NOT NULL DEFAULT 0,
    workdir        TEXT    NOT NULL DEFAULT '',
    core_hours     REAL    NOT NULL DEFAULT 0.0,
    cores          INTEGER NOT NULL DEFAULT 0,   -- ranks x cpus-per-task, as submitted
    exit_reason    TEXT    NOT NULL DEFAULT '',
    remedy         TEXT    NOT NULL DEFAULT '',   -- what the retry ladder tried
    provenance_id  INTEGER REFERENCES provenance(id),
    submitted_at   TEXT,
    finished_at    TEXT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_job_state     ON job(state);
CREATE INDEX IF NOT EXISTS ix_job_stage     ON job(stage, state);
CREATE INDEX IF NOT EXISTS ix_job_structure ON job(structure_id);
CREATE INDEX IF NOT EXISTS ix_job_slurm     ON job(slurm_id);

-- Every gate a structure passed or failed, so `csp status --why` can replay it.
CREATE TABLE IF NOT EXISTS filter_event (
    id             INTEGER PRIMARY KEY,
    structure_id   INTEGER NOT NULL,
    gate           TEXT    NOT NULL,
    passed         INTEGER NOT NULL,
    value          REAL,
    threshold      REAL,
    detail         TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_filter_structure ON filter_event(structure_id);

-- Extracted properties.  `source` distinguishes a computed value from a model:
-- m_dft_raw and m_s_reconstructed are both stored, never merged.
CREATE TABLE IF NOT EXISTS property (
    id             INTEGER PRIMARY KEY,
    structure_id   INTEGER NOT NULL,
    key            TEXT    NOT NULL,
    value          REAL,
    text_value     TEXT    NOT NULL DEFAULT '',
    source         TEXT    NOT NULL,          -- 'dft' | 'mlip' | 'model' | 'mp'
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (structure_id, key, source)
);
CREATE INDEX IF NOT EXISTS ix_property_structure ON property(structure_id);
CREATE INDEX IF NOT EXISTS ix_property_key       ON property(key);

-- The resolved config and code state behind a set of numbers.
CREATE TABLE IF NOT EXISTS provenance (
    id                   INTEGER PRIMARY KEY,
    config_hash          TEXT NOT NULL,
    settings_hash        TEXT NOT NULL DEFAULT '',
    git_sha              TEXT NOT NULL DEFAULT '',
    code_version         TEXT NOT NULL DEFAULT '',
    machine              TEXT NOT NULL DEFAULT '',
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_config_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (config_hash, settings_hash, git_sha)
);

-- Calibration outcomes (Stage 4a/4b), including the fitted alpha/beta used to
-- rescale the screening threshold rather than to rewrite any energy.
CREATE TABLE IF NOT EXISTS calibration (
    id             INTEGER PRIMARY KEY,
    kind           TEXT    NOT NULL CHECK (kind IN ('mp','pilot')),
    n_points       INTEGER NOT NULL,
    mae_e_per_atom REAL,
    mae_e_hull     REAL,
    spearman       REAL,
    volume_drift   REAL,
    alpha          REAL,
    beta           REAL,
    verdict        TEXT    NOT NULL CHECK (verdict IN ('pass','warn','fail')),
    detail         TEXT    NOT NULL DEFAULT '',
    provenance_id  INTEGER REFERENCES provenance(id),
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
