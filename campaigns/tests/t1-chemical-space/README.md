# t1 — source mode `chemical_space`

**You name groups of elements and how many to draw from each**; the source
stage assembles every chemical system those rules allow, then generates
structures for each. You are exploring a space, not a list.

Needs a **GPU** and the **MatterGen checkpoint**.

## What it asks for

The groups here deliberately collapse to exactly **one** system:

```yaml
groups:
  A: {elements: [Ce], pick: 1}
  B: {elements: [Fe], pick: 1}
```

One element, one pick per group → the single system Ce-Fe. A smoke test wants
the whole funnel exercised once, not a sweep.

Within that system, `max_atoms_formula: 3` decides how many stoichiometries get
enumerated, and the count grows faster than it looks:

| `max_atoms_formula` | 2 | 3 | 4 | 6 | 8 |
|---|--:|--:|--:|--:|--:|
| compositions | 1 | 3 | 5 | 11 | 21 |
| structures to generate (×8) | 8 | **24** | 40 | 88 | 168 |

At 3 you get CeFe, CeFe2 and Ce2Fe — 24 generations, three forward passes at
`max_batch_size: 8`. At 8 you would be asking MatterGen for 168, which is not a
smoke test. These numbers were measured by expanding the source stage, not
estimated.

**To test the sweep itself**, widen group A to `[Ce, Sm, Nd]` — that becomes
three systems. Check first that the new systems are covered in the reference
store, or the reference stage will have to compute them and the run stops being
fast:

```bash
python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('/projects/mmi/cspflow-shared/store/index.csv')))
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    by[r['chemsys']][0] += 1
    if r['ready'].strip().lower() == 'true':
        by[r['chemsys']][1] += 1
for s in ('Ce-Fe', 'Fe-Sm', 'Fe-Nd'):
    n, ok = by.get(s, [0, 0]); print(f'{s:<8} {ok}/{n} ready')
PY
```

`max_atoms_formula: 8` caps the reduced formula (CeFe5 is 6) and
`max_rare_earth: 1` stops two rare earths landing in one system.

## Run

```bash
cd campaigns/tests/t1-chemical-space
csp doctor
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 60
csp status
```

## Reset

```bash
rm -rf /scratch/$USER/cspflow/t1-chemical-space report csp-driver-*.log
```
