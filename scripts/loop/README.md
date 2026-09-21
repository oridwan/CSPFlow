# The generate → screen → keep → retry loop

Turns an intuition into a ranked, DFT-worthy shortlist, one round at a time.
It is **not** an autonomous optimiser: it stops at the DFT gate every round and
hands over a command, because submitting compute is yours to do.

```
PROPOSE ──► GENERATE ──► SCREEN ──► MERGE ──► PROMOTE ──► [you submit VASP]
  (agent)    (toolbox)   (GPU/SLURM)          (prepares)           │
     ▲                                                             ▼
     └──────────────── digest.md ◄──── CALIBRATE ◄─────────────────┘
```

## Why each piece is where it is

**The agent runs in exactly one step — `propose`.** Its whole output is a
`plan.yaml` naming generator scripts and their flags. Everything downstream is
deterministic and re-runnable from that file. An agent inside the scoring would
make every round unreproducible, and the result undefendable.

**Screening runs on the GPU partition, never the login node.** Both engines are
torch. Measured on a login-node CPU, one 68-atom Ce₂Fe₁₄B costs ~24 s in
MatterSim at `fmax 0.05` — and about **5 minutes** at the engine's production
default of `fmax 0.01`. A 60-seed round is therefore hours of shared
interactive CPU for work a single GPU finishes in minutes. `screen` submits a
SLURM array; `--local` exists for debugging and says so loudly.

**The screen threshold is parent-relative, and this is not a preference.**
Measured on the store's Ce–Fe–B MatterSim hull:

| phase | E above hull |
|---|---|
| Ce₂Fe₁₄B (mp-4459) | **40.6 meV/atom** |
| Ce₂Fe₁₇ (mp-654) | 44.7 |
| CeFe₅ (mp-11317) | 33.3 |

Ce₂Fe₁₄B is manufactured commercially. MatterSim is a non-magnetic potential
being asked about magnetic intermetallics, and that error does not cancel
between a compound and its elemental references. So an *absolute* cutoff is
meaningless: "under 30 meV" rejects the parent itself. The screenable quantity
is `ddE_hull = E_hull(candidate) − E_hull(parent)`, where the systematic error
largely cancels.

**`ddE_hull` is parent-relative: negative means MORE STABLE THAN THE PARENT,
not below the hull.** Worth stating because misreading it once cost 44
candidates. `Ce3Y(Fe14B)2` at ddE -7.4 meV/atom sits at +33.3 on the full
B-Ce-Fe-Y hull, against the parent's +40.6 -- a real stabilisation, because
Y2Fe14B is only 6.2 meV above the hull while Ce2Fe14B is 40.7.

**A subsystem with no store entry is usually a subsystem with no compounds.**
`hull.missing_subsystems()` reports both cases and cannot tell them apart, so
its output is a FLAG (`hull_incomplete`), never a cut. A hull that truly cannot
place a structure fails loudly as `no_hull` instead. Extending the MLIP hull
needs no DFT: `refstore.py add <CHEMSYS> --apply` then `submit mlip`.

**The three filters do different jobs, and one of them does nothing here.**
Measured on 19 Ce2Fe14B defect structures: `--max-ehull 0.2` removed 0 (the set
spans 0.041-0.156) and `--min-moment 0.458` removed 0 (the lowest is 4.2x the
floor), because every member of a 2:14:1 perturbation set is 82-85% Fe. Those
two are a safety net that earns its keep in the *pivot* batches; the cut that
shortlists is parent-relative `--keep-dd`. The digest prints each filter's
removal count including zeros, so an inert criterion is never quoted as one the
candidates "passed".

**Selection is a Pareto front, not a score.** Scalarising (`0.7·J_s − 0.3·dE`)
hides the trade-off and hands the search a number it can game — the weights
decide the answer before the physics does. The front on
(`ddE_hull`, `J_s_max`) keeps both axes visible and is the figure you would
publish anyway.

**`J_s_max` is an upper bound, and the name is load-bearing.** CHGNet predicts
|m| — magnitudes, no sign — so the sum is the saturation value assuming every
site aligns. For Ce₂Fe₁₄B, where the Fe are parallel, the bound is tight
(1.667 T against our DFT 1.593 T, +4.6 %). For a ferrimagnetic candidate it
reads **high**, which means a search rewarded for large J_s drifts towards
exactly the structures the surrogate gets wrong. That is the loop's main
reward-hacking surface, and `calibrate` is what watches it.

**A pivot batch is mandatory every round.** `generate` refuses a plan whose
batches are all `exploit`. Pure exploitation converges fast onto whatever the
surrogate over-predicts; the ~20 % pivot share is what buys any chance of
leaving the 2:14:1 basin.

## Commands

```bash
L=/projects/mmi/Ridwan/cspflow/Agentic_test/06_Ce2Fe14B_Agentic_test_sept15

python orchestrate.py init --loop $L \
    --parent .../Ce2Fe14B.cif --intuition .../INTUITION.md \
    --elements Ce Fe B --mask-elements Ce --seeds-per-round 60

python orchestrate.py propose  --loop $L            # agent writes plan.yaml
python orchestrate.py generate --loop $L            # runs the toolbox scripts
python orchestrate.py screen   --loop $L            # SLURM array on GPU
python orchestrate.py merge    --loop $L            # union + recompute Pareto
python orchestrate.py promote  --loop $L -k 5       # writes the type-3 campaign, STOPS
python orchestrate.py status    --loop $L
```

`promote` is where the pipeline ends. It writes a complete, `csp doctor`-clean
`structure_list` campaign carrying the reference store's `dft:` block verbatim
[E2], and prints the two commands to run it. **The loop never submits DFT.**

`collect` and `calibrate` exist for measuring CHGNet's drift against finished
DFT, and still work, but are optional -- not part of the generate/filter/propose
path.

**The loop directory must be on shared storage** (`/projects` or `/scratch`).
A run whose directory was in a session-local `/tmp` died in 13 s with no output,
because the compute node could not see it.

`promote` spreads its k picks **along** the front rather than taking the k
highest J_s: the point of calibrating is to find where the surrogate is wrong,
and that needs the cheap end of the front as well as the spectacular end.

## Files

| path | what |
|---|---|
| `hull.py` | the parent-relative MLIP hull; refuses a hull whose only vertices are elements |
| `round.py` | steps 3–10: gate, dedup, relax, ddE_hull, moments, classify, Pareto, digest |
| `round.sbatch` | the GPU array job |
| `orchestrate.py` | round manager: propose / generate / screen / merge / promote / calibrate |
| `<loop>/archive.csv` | every structure ever scored — dedup reads this, so nothing is screened twice |
| `<loop>/structures/*.vasp` | the relaxed geometries, by fingerprint |
| `<loop>/round-NN/digest.md` | **what the round learned** — this is what `propose` reads next |

## What the loop cannot tell you

No SOC anywhere, so K₁, the easy axis, and therefore "is this a permanent
magnet" are **not** answered [D5][G10]. T_c needs J_ij and is out of scope
entirely. CHGNet cannot distinguish FM from ferrimagnetic, so ordering stays a
DFT comparison [D3].

The loop's honest claim is: **it finds high-J_s, hull-plausible,
non-segregating candidates worth spending DFT on.** Not that it finds magnets.

## Stopping

Stop when the budget is spent, when the Pareto front stops moving between
rounds, or — most importantly — when `calibrate` shows the drift **growing**
round on round. A rising drift means the front is being chosen by CHGNet's bias
rather than by the chemistry, and further rounds make it worse, not better.
