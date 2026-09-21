# t2 — source mode `composition_list`

**You name the formulas outright.** No sweep and no element rules: each formula
is expanded over its `z` range and handed to MatterGen to generate structures
for. Use this mode when you already know which phases you care about.

Needs a **GPU** and the **MatterGen checkpoint**.

## What it asks for

Two real Ce-Fe phases, 8 structures each at z = 1:

| formula | atoms at z=1 | generated |
|---|--:|--:|
| CeFe2 | 6 | 8 |
| CeFe5 | 6 | 8 |

Both exist in the reference store, so a good generated structure should land
near `e_above_hull = 0` — a soft check on the generator, not just the plumbing.

`from_file: inputs/compositions.csv` is the other way to feed this mode; it is
left out so this campaign.yaml is the single source of truth.

## How it differs from t1

Only the source stage. Same engines, same chemistry, same DFT. Running both and
getting the same behaviour after the source stage is the point of having two.

## Run

```bash
cd campaigns/tests/t2-composition-list
csp doctor
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 60
csp status
```

MatterGen on a GPU is the long pole — roughly 10-30 min for a batch this small,
plus the GPU queue. `timeout_per_batch: 900` means a batch that stalls is
abandoned rather than holding a GPU.

## Reset

```bash
rm -rf /scratch/$USER/cspflow/t2-composition-list report csp-driver-*.log
```
