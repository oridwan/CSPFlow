# 09 — The DFT recipe

> **Guide 09 of 13** · Previous: [08 — Running on your cluster](08-machines.md) · [Guide home](README.md) · Next: [10 — The reference set](10-reference-set.md)

`recipe.yaml` is the ladder each surviving structure climbs: a list of VASP
steps, each with its own INCAR, k-points, resources and failure handling.
`csp init` copies one into your campaign folder; `dft.recipe` in
`campaign.yaml` points at it.

```bash
csp recipe            # this campaign's ladder, fully resolved
csp recipe magnets    # a shipped recipe by name
```

`csp recipe` prints every tag literal, with nothing deferred — there is no
value in the output whose meaning requires knowing pymatgen to predict.

## Shape

```yaml
name: magnets

stages:
  - name: relax
    incar:
      PREC:   Accurate
      ENCUT:  520
      ALGO:   Fast
      EDIFF:  1.0e-4
      NELM:   200
      ISPIN:  2
      LASPH:  .TRUE.
      LORBIT: 11
      IBRION: 1
      ISIF:   3
      NSW:    99
      EDIFFG: -0.01
      ISMEAR: 1
      SIGMA:  0.05
      NCORE:  8
    kpoints:   {scheme: reciprocal_density, value: 64}
    resources: {role: cpu, ntasks: 64, time: "24:00:00"}
    retry:
      - when: timeout
        remedy: resume_from_contcar
      - when: scf_not_converged
        set: {ALGO: Normal, NELM: 300}
      - when: ionic_step_limit
        set: {NSW: 200}
        remedy: resume_from_contcar

  - name: static
    inherit: relax          # copies every tag above, then overrides
    incar:
      IBRION: -1
      NSW:    0
      ISMEAR: -5            # tetrahedron for the final energy and DOS
      EDIFF:  1.0e-6
      LCHARG: .TRUE.
    resources: {role: cpu, ntasks: 64, time: "12:00:00"}
```

Each step starts from the previous step's **relaxed** geometry, so the ladder
above means "relax, then take the final energy on a tetrahedron mesh."

## `incar` — free-form, with two guardrails

Add any tag you like. An unrecognised tag is **written as given and warned
about, never dropped** — refusing one would make "add any tag you like" false.

There is no implicit base set. No `MPRelaxSet`, nothing inherited from
pymatgen. A base set is *library version state* rather than campaign state: an
upgrade would move `ENCUT` or the LDA+U tables mid-campaign, and provenance
recording `base_set: MPRelaxSet` would reproduce nothing.

That leaves one obligation, which cspflow enforces. **A recipe that omits a tag
VASP does not default sensibly is refused**, with the reason:

| tag | what VASP would do instead |
|---|---|
| `ENCUT` | use `max(ENMAX)` over the POTCARs — which **changes with composition** (measured 267.9 eV for Sm_3/Fe/Ti, 295.4 eV once Cu is present). A hull built on that compares incomparable numbers, and the error does not cancel in a formation energy |
| `ISPIN` | 1, non-spin-polarised: every moment would be zero |
| `LASPH` | `.FALSE.`, turning off aspherical corrections that matter for 4f |
| `NELM` | 60 — the SCF gives up early and the job **exits successfully** having not converged |
| `LORBIT` | unset: no site-projected moments, so the analysis stage has nothing to read |

Some tags are computed per structure rather than written here, and setting them
explicitly still wins: `MAGMOM`, `NBANDS`, `SYSTEM`, `LMAXMIX`, and the `LDAU*`
family. `LMAXMIX` is derived from the actual POTCARs (6 with 4f in valence, 4
for d-only chemistries), which is more reliable than asking you to keep it in
step with `dft.rare_earth.f_treatment`.

> **Write INCAR booleans as `.TRUE.` / `.FALSE.` strings.** YAML 1.1 turns bare
> `yes`/`no`/`on`/`off` into booleans. `LASPH: true` is accepted and converted,
> but the VASP spelling is what the file should say.

## `inherit`

`inherit: relax` copies every tag from that step, then applies this step's
`incar` on top. It is a **copy**, not a reference: `csp recipe` shows the
resolved result, so no one has to hold the chain in their head.

## `kpoints`

```yaml
kpoints: {scheme: reciprocal_density, value: 64}
```

| scheme | `value` | meaning |
|---|---|---|
| `reciprocal_density` | a density | grid chosen so k-point density in reciprocal space is constant across cells of different size — the one to use for a hull |
| `kspacing` | Å⁻¹ | target spacing; VASP's own `KSPACING` semantics |
| `explicit` | `[n1, n2, n3]` | a literal grid |

Grids are Γ-centred, which preserves the crystal's point symmetry.

`reciprocal_density` is the default because a hull compares energies across
cells of very different size, and a fixed grid does not give them comparable
sampling.

## `resources`

Per step, so a cheap static does not book the same nodes as the relax:

```yaml
resources: {role: cpu, ntasks: 64, time: "24:00:00"}
```

`role` maps to a partition in [`machine.yaml`](08-machines.md). Omitted keys fall
back to that machine's defaults.

**These two keys are free to change.** `recipe_id` — the hash that decides
whether two calculations may share a convex hull — covers the recipe's `name:`
and its **physics**: every INCAR tag and the k-point setting. It does **not**
cover `ntasks`, `time` or `mem`.

That asymmetry is useful and worth knowing:

* to make jobs cheaper or faster to schedule, change `ntasks` and `time` as
  much as you like. A 6-atom cell does not need 64 ranks, and a 16-rank job
  also starts sooner.
* to change any INCAR tag or the k-point density, understand that you have
  created a **new energy scale**. Nothing computed before is comparable with
  anything computed after, and the hull will mix them silently.

The trap runs the other way too: **renaming a copied recipe invents a mismatch
that is not real.** A file that is byte-for-byte `magnets` but called
`magnets-fast` hashes differently, and `csp doctor` will report it as a scale
mismatch against the reference store. Keep `name:` unless the physics changed.

> **`role: cpu` is not decoration on this cluster.** The `cpu` role in
> `orion.yaml` carries a `constraint` that excludes nodes without AVX-512; our
> `vasp_std` is compiled with it and dies with `illegal instruction` before
> writing an OUTCAR on a node that lacks it. The `short` role carries no such
> constraint. Keep DFT steps on `cpu`.

## `retry` — the ladder

A list of rules tried in order. Each has a **trigger** and a **response**:

```yaml
retry:
  - when: timeout
    remedy: resume_from_contcar
  - when: scf_not_converged
    set: {ALGO: Normal, NELM: 300}
  - when: out_of_memory
    set: {NCORE: 4}
```

| trigger | raised when |
|---|---|
| `timeout` | the scheduler killed it at the walltime |
| `out_of_memory` | the scheduler killed it for memory |
| `scf_not_converged` | VASP exited cleanly without reaching the electronic criterion |
| `ionic_step_limit` | it exited cleanly having used every `NSW` step |

| response | effect |
|---|---|
| `set: {TAG: value}` | INCAR overrides for the retry |
| `remedy: resume_from_contcar` | restart from the CONTCAR rather than the original POSCAR |

The two are independent and often used together — a timeout wants a resume, an
SCF failure wants a different algorithm, and they are **not interchangeable**.
Matching the response to the actual cause is the whole point of keying on a
trigger rather than retrying blindly.

A structure gets one attempt per rung, plus the original. Exit code 127
(`command not found`) is **never** retried: this account hit it 287 times in 60
days, and retrying it unchanged burns a submission slot to reproduce the
identical failure.

> **Write `when:`, never `on:`.** YAML 1.1 parses a bare `on` as the boolean
> `True`, so `- on: timeout` becomes `{True: 'timeout'}` and your trigger
> silently vanishes. cspflow warns if it finds a retry rule with no `when:`.

## Changing it safely

```bash
csp recipe > /tmp/before.txt
$EDITOR recipe.yaml
csp recipe > /tmp/after.txt && diff /tmp/before.txt /tmp/after.txt
```

The tags most often worth changing are `ENCUT`, `NSW`, `EDIFFG` and `NCORE`.
For a one-off, `dft.incar_overrides` in `campaign.yaml` applies on top of
**every** step without touching the recipe at all.


---

## See also

* [Every setting in `campaign.yaml`](06-settings.md) — the `dft:` block that points here
* [Running on your cluster](08-machines.md) — what `role:` maps to
* [The reference set](10-reference-set.md) — why `recipe_id` decides what may share a hull
