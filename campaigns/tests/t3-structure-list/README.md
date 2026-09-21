# t3 — source mode `structure_list`

**You supply the structures.** There is no generation stage at all: the seeds
go straight to MLIP screening. This is the mode the real CePdGe and CeFeB
campaigns use.

**Run this one first.** Shortest path from `csp run` to a finished VASP job.

## The seeds

`inputs/seeds/` holds the DFT-relaxed CONTCARs of three real Ce-Fe phases,
copied from the reference store:

| file | phase | atoms |
|---|---|--:|
| `mp-204-CeFe2.vasp` | CeFe2 | 6 |
| `mp-11317-CeFe5.vasp` | CeFe5 | 6 |
| `mp-1213958-Ce4Fe.vasp` | Ce4Fe | 10 |

They were chosen so the test has a **known expected answer**: all three are on
or near the hull, so all three should survive the filter and return
`e_above_hull` near 0. Ce's `m_dft_raw` near 0 is also expected — the 4f
electrons are in the POTCAR core.

## Run

```bash
cd campaigns/tests/t3-structure-list
csp doctor --elements Ce,Fe
sbatch /projects/mmi/Ridwan/cspflow/scripts/campaign_driver.sbatch . "" 60
csp status
```

Expect 3 seeds in, 3 structures to DFT, 3 jobs in `squeue` named
`t3-structure-list-000<n>-...`, each relax+static in one job.

## Reset

```bash
rm -rf /scratch/$USER/cspflow/t3-structure-list report csp-driver-*.log
```
