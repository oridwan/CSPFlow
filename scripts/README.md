# scripts/

Operational scripts that sit beside the `csp` command. They are not part of the
installed package. Run them with the `cspflow` environment active, from the
repository root or with the full path.

The top level holds what a group member runs in normal use. The subfolders hold
tools you reach for rarely, usually after something has gone wrong. Every
script that changes anything is a **dry run by default** and needs `--apply` to
write.

## Top level: everyday use

| script | what it is for |
|---|---|
| `build_env.sh` | Builds the one conda environment (MatterGen + MatterSim + cspflow). `env.lock.txt` is the exact package list it last produced. |
| `campaign_driver.sbatch` | Runs `csp run` for a campaign as its own SLURM job, so the driver does not live on a login node. Arguments: campaign folder, stage slice (default `--through reference`), seconds between cycles. From inside a campaign: `sbatch <repo>/scripts/campaign_driver.sbatch . "--from filter" 300` |
| `campaign_audit.py` | Shows what a campaign is actually doing, by reconciling the database, SLURM and the files on disk, which can disagree. Run it before any repair. |
| `requeue.py` | Puts structures back in the DFT queue, chosen by id, by failure reason, or by dead job. `--from-seed` discards on-disk work and restarts from the seed. |
| `refstore.py` | The shared reference-store tool: `status`, `add`, `submit`, `index`, `hull`. See [docs/10-reference-set.md](../docs/10-reference-set.md). |
| `refstore_dft.sbatch`, `refstore_mlip.sbatch` | The jobs `refstore.py submit` sends. `refstore.py` finds them **beside itself**, so these three files always move together. |

## `repair/`: campaign surgery

One-off fixes for a campaign whose database and files have drifted apart. Each
script's header says which decision (Dxxx) it came from and when to use it.

| script | use it when |
|---|---|
| `reparse_dft.py` | The DFT *parser* changed and finished runs need re-reading. |
| `reopen_for_dft.py` | Structures were filtered out on hull distance and should go to DFT after all. |
| `ingest_new_seeds.py` | You added seed files to a campaign that already ran. `csp run` ignores them silently. |
| `exclude_structures.py` | A structure must be cancelled and removed for good (`scancel` alone is undone by the driver). |
| `rebuild_campaign.py` | `campaign.db` is wrong or lost, and must be rebuilt from the per-composition databases. |
| `recarry_relaxed_geometry.py` | Relaxed MLIP cells were lost to the float32 bug (D145). Run it once on campaigns screened before the fix. |

## `store-maintenance/`: reference-store admin

| script | what it is for |
|---|---|
| `refcheck.py` | Explains why store structures are unfinished, bucketed by VASP's own error. |
| `refhealth.py` | Checks whether each **running** store job is still making progress. |
| `shared_store_transfer.sbatch` | Copies a store to another location with group permissions intact, on the DTN partition. |

## `loop/`: the intuition loop

A separate research tool with its own [README](loop/README.md).
