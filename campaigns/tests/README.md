# Three test campaigns, one per source mode

Small, fast, throwaway campaigns whose only job is to prove the pipeline works
end to end on this cluster. Each exercises a different way structures get INTO
the funnel; everything after the source stage is identical between them, on
purpose, so a difference in outcome points at the source mode and nothing else.

| folder | source mode | GPU? | MatterGen? | into the funnel | run it |
|---|---|:--:|:--:|---|---|
| `t3-structure-list` | you supply the structures | screen only | no | **3 seeds**, read from disk | **first** |
| `t2-composition-list` | you name the formulas | yes | yes | 2 compositions -> **16 generated** | second |
| `t1-chemical-space` | you name element groups, it derives the formulas | yes | yes | 3 compositions -> **24 generated** | third |

Those counts are measured, by expanding each source stage in a throwaway
workdir, not estimated. All three end `csp doctor` with **All checks passed**.

**Run `t3` first.** It is the only one with no generation stage, so it is the
shortest path from `csp run` to a finished VASP job. If the machine profile,
the POTCARs, the reference store or the new one-job-per-structure submission
are wrong, `t3` tells you in minutes and the other two would only tell you
after a GPU queue wait.

## The chemistry, and why it is the same in all three

**Ce-Fe.** All five of its phases are already in the shared reference store at
our own DFT settings (`ready=True`): CeFe2, CeFe5, Ce4Fe and two Ce2Fe17. So
the reference stage finds a complete hull and computes nothing — the stage that
would otherwise dominate a test run costs seconds rather than days of VASP.

It is also magnetic RE-TM chemistry with Ce, so these tests exercise the same
frozen-4f path the real campaigns use, rather than a simplified one.

## What makes them fast — and what deliberately does NOT

Fast comes from **small** and **few**, never from **different**:

| made small | left exactly as production |
|---|---|
| cells: 6-10 atoms, `z: [1,1]`, `max_atoms: 12` | every INCAR tag, including ENCUT 520 |
| counts: 8 structures generated, `max_total: 6` DFT | the k-point density (64) |
| `ntasks` 64 -> 16 and walltimes 24h -> 4h | `ISPIN 2`, `LORBIT 11`, `strict: true` |
| MLIP `fmax` 0.01 -> 0.05, `max_steps` -> 100 | the whole retry ladder |

**Why the INCAR is untouched.** `recipe_id` hashes the recipe's `name:` and its
physics; it does **not** hash `ntasks` or `time`. Shrinking the allocation
therefore costs nothing, while changing ENCUT or the k-point density would put
candidate energies on a different scale from the reference hull. That hull
still builds, still looks correct, and ranks wrongly (D101). `csp doctor`
reports `matches the store (6e44a4bf351d98a9) -- the hull is on one scale`, and
it must keep saying that.

So: **do not edit `recipe.yaml`'s `name:` or any INCAR tag.** Change `ntasks`
and `time` freely.

> Worth knowing: the production `CePdGe` campaign does **not** currently match
> the store — it uses `ferri`/`ferrimagnetic_retm` against the store's
> `ferro`/`ferromagnetic`, and `csp doctor` warns about it there. These test
> campaigns copy the store's `dft:` block verbatim instead, because a test whose
> numbers cannot be graded is not a test. That production mismatch is a separate
> question and is not resolved here.

## How to grade the result

`t3` has a known answer. Its three seeds are the store's own DFT-relaxed
CONTCARs, so they sit on or very near the hull by construction:

* all three should survive the filter;
* `e_above_hull` should come back near **0**;
* `m_dft_raw` on Ce should be near **0** — that is correct, not a failure. The
  4f electrons are frozen into the POTCAR core, so the computed moment lives on
  Fe and the Ce contribution is reconstructed afterwards (`m_s_reconstructed`).

`t1` and `t2` generate new structures, so there is no exact expected number —
what you are checking there is that generation runs, that structures reach DFT,
and that the funnel counts add up.

## Running one

```bash
cd campaigns/tests/t3-structure-list

csp doctor                    # read-only; must end "All checks passed."
csp doctor --elements Ce,Fe   # also checks POTCARs and the magnetism table

# submit the driver as its own batch job ("" = run the whole funnel,
# 60 = seconds between cycles)
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 60

# watch it
csp status
squeue --me
tail -f csp-driver-*.log

# when it finishes
csp report                    # writes report/report.html + candidates.csv
python /projects/mmi/Ridwan/cspflow/scripts/campaign_audit.py --campaign .
```

Jobs appear in `squeue` named after the structure they are running, one job per
structure, e.g. `t3-structure-list-0002-CeFe5-mp_11317_CeFe5`.

## Starting over

Everything a run creates lives in `workdir`, not here:

```bash
rm -rf /scratch/$USER/cspflow/t3-structure-list
rm -f  campaigns/tests/t3-structure-list/csp-driver-*.log
rm -rf campaigns/tests/t3-structure-list/report
```

The campaign folder itself (`campaign.yaml`, `recipe.yaml`, `machine.yaml`,
`inputs/`) is input only and is never written to. Deleting the workdir is a
true reset.

**Cancel anything still queued first** — `scancel --me --name=t3-structure-list-*`
does not work, so use `squeue --me -o "%i %j"` and cancel by id, or
`scancel --me` if nothing else of yours is running.
