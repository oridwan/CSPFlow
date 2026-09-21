# ---------------------------------------------------------------------------
# orchestrate.py -- the outer loop: PROPOSE -> GENERATE -> SCREEN -> PROMOTE
#                   -> (you submit the DFT) -> CALIBRATE -> propose again.
#
# WHAT THIS IS AND IS NOT
#   It is a round manager over an append-only archive. Each round is ordinary
#   cspflow/toolbox work; this file decides WHAT to try next and WHAT to spend
#   DFT on, and records both so the decision is auditable afterwards.
#
#   It is NOT an autonomous optimiser. It stops at the DFT gate every round and
#   hands over a command, because submitting compute is the user's call here.
#   `calibrate` picks the loop back up once those jobs land.
#
# THE AGENT IS IN EXACTLY ONE STEP
#   `propose` is the only place a model runs, and its entire output is a
#   plan.yaml naming generator scripts and their parameters. Everything after
#   that is deterministic and re-runnable from that file. If the agent were also
#   in the scoring, no round could be reproduced and no result defended.
#
# THE FEEDBACK THAT MAKES IT A LOOP RATHER THAN A BATCH
#   `propose` is handed the previous round's digest.md, which carries: which
#   descriptor actually tracked J_s, which routes reached the Pareto front,
#   which gates bound, and the surrogate's measured drift against DFT. Without
#   that the loop is just N independent batches with extra steps.
#
# THE BUDGET SPLIT, AND WHY IT IS NOT ALL EXPLOITATION
#   Each round's proposal is asked to divide its seeds roughly
#       ~50% EXPLOIT  enumerate the neighbourhood of a Pareto winner
#       ~30% WIDEN    same route, larger x / supercell / more species
#       ~20% PIVOT    a different route, or a prototype hop (1:12, 2:17)
#   A pure-exploit loop converges fast onto whatever the surrogate over-predicts.
#   The pivot share is what buys the chance of leaving the 2:14:1 basin at all.
#
# COMMANDS
#   init       create the loop directory and write loop.yaml
#   propose    run the agent -> round-NN/plan.yaml     (the only model call)
#   generate   execute plan.yaml -> round-NN/seeds/
#   screen     submit the screen to the GPU partition as a SLURM array
#   merge      combine the array tasks' shards, recompute the Pareto front
#   promote    pick k from the Pareto front, WRITE a ready cspflow DFT
#              campaign, print the submit command, and STOP
#   collect    read the finished VASP runs into results.csv
#   calibrate  read the finished DFT and measure surrogate drift
#   status     where the loop is
#
# RUN
#   python orchestrate.py init --loop loop/ --parent Ce2Fe14B.cif \
#       --intuition INTUITION.md --elements Ce Fe B
#   python orchestrate.py propose  --loop loop/       # agent writes plan.yaml
#   python orchestrate.py generate --loop loop/
#   python orchestrate.py screen   --loop loop/
#   python orchestrate.py promote  --loop loop/ -k 5  # prepares DFT, stops
#   # ... you submit, jobs finish ...
#   python orchestrate.py calibrate --loop loop/
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse, csv, json, os, shlex, shutil, subprocess, sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SKILLS = Path(os.environ.get("CSPFLOW_SKILLS", "/users/oridwan/.claude/skills"))
TOOLBOX = SKILLS / "re-substitution-magnets" / "scripts"


def _cfg(loop: Path) -> dict:
    f = loop / "loop.yaml"
    if not f.is_file():
        sys.exit(f"{f} not found -- run `init` first")
    return yaml.safe_load(f.read_text())


def _round_dirs(loop: Path):
    return sorted(loop.glob("round-*"), key=lambda p: p.name)


def _current(loop: Path) -> int:
    d = _round_dirs(loop)
    return int(d[-1].name.split("-")[1]) if d else 0


# --------------------------------------------------------------------------
def cmd_init(a):
    loop = Path(a.loop)
    loop.mkdir(parents=True, exist_ok=True)
    cfg = {
        "parent": str(Path(a.parent).resolve()),
        "intuition": str(Path(a.intuition).resolve()) if a.intuition else None,
        "elements": a.elements,
        "magnetic_element": a.magnetic_element,
        "mask_elements": a.mask_elements,
        "keep_ehull_meV": a.keep_ehull,
        "max_ehull_eV": 0.2,
        "min_moment_muB": 0.458,
        "seeds_per_round": a.seeds_per_round,
        "dft_per_round": a.dft_per_round,
        "budget_split": {"exploit": 0.5, "widen": 0.3, "pivot": 0.2},
        "machine": a.machine,
    }
    (loop / "loop.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {loop}/loop.yaml")

    # CONTEXT.md, computed not asked for. `propose` is told to read it and NOT
    # to re-derive any structural fact, which only works if it is actually here
    # -- the same deterministic-crystallography-first rule intuit_to_seeds.sh
    # follows. Without it the agent silently falls back to deriving orbits
    # itself, which is the one thing the design exists to prevent.
    ctx = [f"# CONTEXT.md -- computed crystallography of {Path(a.parent).name}", "",
           "Generated by orchestrate.py with pymatgen + spglib. Every number below",
           "is measured from the structure file. Do not re-derive these; read them.",
           ""]
    for title, script in (("Orbits, layers, coordination, Hill check", "analyze_cif.py"),
                          ("Coordination shell resolved by orbit", "orbit_detail.py")):
        tool = TOOLBOX / script
        if not tool.is_file():
            continue
        r = subprocess.run([sys.executable, str(tool), cfg["parent"]],
                           capture_output=True, text=True)
        ctx += [f"## {title}", "```", r.stdout.rstrip(), "```", ""]
    (loop / "CONTEXT.md").write_text("\n".join(ctx))
    print(f"wrote {loop}/CONTEXT.md  ({len(ctx)} lines of computed crystallography)")
    print(yaml.safe_dump(cfg, sort_keys=False))


# --------------------------------------------------------------------------
PROPOSE_PROMPT = """\
You are proposing ONE round of a generate/screen loop. Your entire output is a
plan.yaml. You do not run generators, you do not score anything, and you do not
write coordinates.

READ, in this order:
  * {intuition}          -- the scientist's intuition, unchanged
  * {context}            -- the parent's computed crystallography. Authoritative.
  * {digest}             -- what the LAST round learned. THIS IS THE POINT.
  * {rules}              -- RULES.md; section G binds a moment-density campaign
  * {archive}            -- every structure tried so far (do not re-propose one)

THEN write {plan} and nothing else. Schema:

  round: <int>
  rationale: |
    Two or three sentences. What the last digest told you, and what you are
    therefore changing. If the digest said J_s tracks n_Fe/V and not x_Fe at.%,
    say how this plan acts on that.
  batches:
    - tag: <short-name>
      intent: exploit | widen | pivot
      why: <one line -- which Pareto point or which finding this comes from>
      script: enumerate_defect_configs.py | enumerate_orbit_substitutions.py |
              enumerate_interstitials.py | partial_sublattice_replace.py |
              deform_series.py
      args: {{ ... exactly the flags that script takes ... }}

RULES THAT BIND THE PLAN
  * total seeds across batches: about {seeds} (the screen costs ~35 s each)
  * split the budget roughly {split} across exploit / widen / pivot
  * a PIVOT batch is mandatory every round -- a plan that only exploits is
    rejected. It is what stops the loop converging on surrogate error.
  * do NOT re-propose anything already in archive.csv
  * cite rule ids where a choice is forced by one
  * if the last digest's Pareto front was EMPTY, do not simply widen --keep-ehull.
    Say why the axis is exhausted and pivot.
"""


def cmd_propose(a):
    loop = Path(a.loop)
    cfg = _cfg(loop)
    n = _current(loop) + 1
    rdir = loop / f"round-{n:02d}"
    rdir.mkdir(exist_ok=True)
    prev = loop / f"round-{n-1:02d}" / "digest.md"
    prompt = PROPOSE_PROMPT.format(
        intuition=cfg.get("intuition") or "(none given)",
        context=loop / "CONTEXT.md",
        digest=prev if prev.is_file() else "(first round -- no digest yet)",
        rules=SKILLS / "re-substitution-magnets" / "RULES.md",
        archive=loop / "archive.csv",
        plan=rdir / "plan.yaml",
        seeds=cfg["seeds_per_round"],
        split=cfg["budget_split"],
    )
    (rdir / ".propose_prompt.txt").write_text(prompt)
    if a.dry_run:
        print(f"dry run -- wrote {rdir}/.propose_prompt.txt")
        return
    claude = _claude_bin()
    if not claude:
        sys.exit("claude binary not found; set CLAUDE_BIN or use --dry-run")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_AGENT"))}
    print(f"proposing round {n} ...")
    with open(rdir / "propose.log", "w") as log:
        subprocess.run(
            [claude, "-p", prompt, "--max-turns", "25",
             "--allowedTools", "Read,Write,Glob,Grep,Bash(python*),Bash(head*),Bash(cat*)",
             "--permission-mode", "acceptEdits",
             "--output-format", "stream-json", "--verbose"],
            cwd=rdir, stdout=log, stderr=subprocess.STDOUT, env=env)
    plan = rdir / "plan.yaml"
    if not plan.is_file():
        sys.exit(f"the agent wrote no plan.yaml -- see {rdir}/propose.log")
    print(plan.read_text())


def _claude_bin():
    for c in (os.environ.get("CLAUDE_BIN"),):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    from shutil import which
    if which("claude"):
        return which("claude")
    cands = sorted(Path("/users/oridwan/.local/share/code-server/extensions").glob(
        "anthropic.claude-code-*-linux-x64/resources/native-binary/claude"))
    return str(cands[-1]) if cands else None


# --------------------------------------------------------------------------
def cmd_generate(a):
    loop = Path(a.loop)
    cfg = _cfg(loop)
    n = a.round or _current(loop)
    rdir = loop / f"round-{n:02d}"
    plan = yaml.safe_load((rdir / "plan.yaml").read_text())
    seeds = rdir / "seeds"
    seeds.mkdir(exist_ok=True)
    intents = [b.get("intent") for b in plan.get("batches", [])]
    if "pivot" not in intents:
        sys.exit("plan has no PIVOT batch. A plan that only exploits is rejected "
                 "-- it is how the loop converges on surrogate error. Re-propose.")
    for b in plan["batches"]:
        script = TOOLBOX / b["script"]
        if not script.is_file():
            script = SKILLS / "matsci-agent-pipeline" / "scripts" / b["script"]
        out = seeds / b["tag"]
        cmd = [sys.executable, str(script)]
        for k, v in b["args"].items():
            flag = f"--{k.replace('_','-')}"
            if v is True:
                cmd.append(flag)
            elif isinstance(v, (list, tuple)):
                cmd += [flag] + [str(x) for x in v]
            else:
                cmd += [flag, str(v)]
        cmd += ["--outdir", str(out)]
        print(f"\n=== {b['tag']} [{b.get('intent')}] ===\n{' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.run(cmd, check=False)
    made = list(seeds.rglob("*.vasp"))
    print(f"\ngenerated {len(made)} seeds under {seeds}")


# --------------------------------------------------------------------------
def cmd_screen(a):
    """Submit the round's screen to the GPU partition as a SLURM array.

    NOT run inline. Both engines are torch, and a 60-seed round is hours of CPU
    -- that does not belong on a login node, which is shared and interactive.
    `--local` exists for a handful of structures while debugging; it says so
    loudly, because it is the wrong way to run a real round.
    """
    loop = Path(a.loop).resolve()
    cfg = _cfg(loop)
    n = a.round or _current(loop)
    rdir = loop / f"round-{n:02d}"
    seeds = str(rdir / "seeds" / "**" / "*.vasp")
    nseeds = len(list((rdir / "seeds").rglob("*.vasp")))
    if not nseeds:
        sys.exit(f"no seeds under {rdir/'seeds'} -- run `generate` first")

    extra = ["--parent", cfg["parent"],
             "--keep-ehull", str(cfg["keep_ehull_meV"]),
             "--max-ehull", str(cfg.get("max_ehull_eV", 0.2)),
             "--min-moment", str(cfg.get("min_moment_muB", 0.458)),
             "--magnetic-element", cfg["magnetic_element"]]
    if cfg.get("elements"):
        extra += ["--elements"] + list(cfg["elements"])
    if cfg.get("mask_elements"):
        extra += ["--mask-elements"] + list(cfg["mask_elements"])

    if a.local:
        print("!! --local: running on THIS node. Fine for a few structures while\n"
              "!! debugging, wrong for a real round -- use SLURM.\n")
        subprocess.run([sys.executable, str(HERE / "round.py"), seeds,
                        "--archive", str(loop), "--round", str(n)] + extra, check=False)
        return

    chunk = a.chunk
    ntasks = (nseeds + chunk - 1) // chunk
    env = dict(os.environ,
               LOOP_DIR=str(loop), LOOP_ROUND=str(n), LOOP_SEEDS=seeds,
               LOOP_CHUNK=str(chunk), LOOP_SCRIPTS=str(HERE),
               LOOP_ARGS=" ".join(shlex.quote(x) for x in extra))
    cmd = ["sbatch", "--parsable", f"--array=0-{ntasks-1}",
           f"--output={rdir}/screen-%A_%a.out", str(HERE / "round.sbatch")]
    print(f"round {n}: {nseeds} seeds -> {ntasks} array task(s) of {chunk}, GPU partition")
    print(" ".join(shlex.quote(c) for c in cmd))
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sbatch failed: {r.stderr.strip()}")
    jid = r.stdout.strip()
    (rdir / "screen.jobid").write_text(jid)
    print(f"submitted {jid}\n  squeue -j {jid}\n  then: orchestrate.py merge --loop {loop}")


def cmd_merge(a):
    """Concatenate the array tasks' shards into the round's single result.

    Array tasks each write scored.tN.csv / pareto.tN.csv; the Pareto front must
    be recomputed over the UNION, because a point that is non-dominated inside
    one shard can be dominated by a point in another.
    """
    sys.path.insert(0, str(HERE))
    from round import pareto_front, _write
    loop = Path(a.loop).resolve()
    n = a.round or _current(loop)
    rdir = loop / f"round-{n:02d}"
    shards = sorted(rdir.glob("scored.t*.csv"))
    if not shards:
        sys.exit(f"no shards in {rdir} -- has the array finished? "
                 f"check {rdir}/screen-*.out")
    rows = []
    for f in shards:
        rows += list(csv.DictReader(open(f)))
    # csv.DictReader returns every field as a string. Coercing only the two
    # columns the Pareto step needs left the digest's tables formatting strings
    # with "%.1f" -- so coerce EVERY numeric-looking field once, here, and let
    # the rest of the pipeline see the same types round.py produced.
    NEVER_NUMERIC = {"file", "formula", "status", "gate", "fingerprint",
                     "topology_class", "shard"}
    for r in rows:
        for k, v in list(r.items()):
            if k in NEVER_NUMERIC:
                continue
            if v in (None, "", "None"):
                r[k] = None
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        r["alpha_flag"] = bool(r.get("alpha_flag"))
        r["hull_incomplete"] = bool(r.get("hull_incomplete"))
    # CROSS-SHARD DEDUP. Each array task dedups against archive.csv, which is
    # only written when that task FINISHES -- so concurrent tasks never see each
    # other's structures and an identical cell screened in two shards survives
    # twice. Measured on this round: within-shard dedup caught 13, and 20
    # duplicate fingerprints still came through, putting the same antisite on the
    # Pareto front twice. merge is the first step that sees everything, so it is
    # where the union is deduplicated.
    scored_all = [r for r in rows if r.get("status") == "scored"]
    seen_fp, scored = set(), []
    for r in scored_all:
        fp = r.get("fingerprint")
        if fp and fp in seen_fp:
            r["status"] = "duplicate"
            r["gate"] = f"cross-shard duplicate of {fp}"
            continue
        if fp:
            seen_fp.add(fp)
        scored.append(r)
    n_xdup = len(scored_all) - len(scored)

    cfg = _cfg(loop)
    MAG = cfg["magnetic_element"]
    # Recount the filters over the MERGED set. Passing a=None left the digest's
    # filter table empty, which is precisely the table that exists to show when a
    # criterion did no work.
    # merge MUST apply the SAME filter set as round.py, or the merged shortlist
    # is not the union of the shards' shortlists. Until 2026-09-15 it applied
    # only alpha_flag and the ceiling: the moment floor and the absolute E_hull
    # floor were LABELS in the table, never evaluated, so merge could keep a
    # structure round.py had cut. The thresholds come from loop.yaml so the two
    # paths cannot drift apart again.
    max_ehull = float(cfg.get("max_ehull_eV", 0.2))
    min_moment = float(cfg.get("min_moment_muB", 0.458))
    cuts = {f"E_hull >= {max_ehull:.2f} eV/atom": 0,
            f"moment <= {min_moment:.3f} muB/atom": 0,
            f"alpha-{MAG} segregation [G2]": 0,
            f"E_hull > {cfg['keep_ehull_meV']:.0f} meV/atom (shortlist)": 0}
    passed = []
    for r in scored:
        eh = r.get("E_hull_meV")
        if eh is None or eh / 1000.0 >= max_ehull:
            cuts[f"E_hull >= {max_ehull:.2f} eV/atom"] += 1; continue
        m = r.get("muB_per_atom")
        m = float(m) if m not in (None, "") else None
        if m is None or m <= min_moment:
            cuts[f"moment <= {min_moment:.3f} muB/atom"] += 1; continue
        if r.get("alpha_flag"):
            cuts[f"alpha-{MAG} segregation [G2]"] += 1; continue
        passed.append(r)
    kept = [r for r in passed if r["E_hull_meV"] is not None
            and r["E_hull_meV"] <= cfg["keep_ehull_meV"]]
    cuts[f"E_hull > {cfg['keep_ehull_meV']:.0f} meV/atom (shortlist)"] = len(passed) - len(kept)
    front = pareto_front(kept, "E_hull_meV", "J_s_max_T")

    class _A:                       # what _digest_tables reads off `a`
        filter_cuts = cuts
        keep_ehull = cfg["keep_ehull_meV"]
    _write(rdir / "scored.csv", rows)
    _write(rdir / "pareto.csv", front)

    # Regenerate the digest over the UNION. `propose` reads digest.md and nothing
    # else, so leaving only per-shard digests would feed the next round one
    # task's worth of evidence and call it the round -- and the correlation
    # table in particular is meaningless on a 40-structure slice of a 400-
    # structure round.
    from round import _digest_tables
    head = []
    first = sorted(rdir.glob("digest.t*.md"))
    if first:
        for line in first[0].read_text().splitlines():
            if line.startswith("Parent reference:"):
                head.append(line)
    body = [f"# Round {n} digest (merged over {len(shards)} shard(s))", "",
            f"- scored: {len(scored)} unique ({n_xdup} cross-shard duplicates removed)"
            f"   kept (E_hull <= {cfg['keep_ehull_meV']:g} meV/atom, "
            f"no alpha flag): {len(kept)}   Pareto front: {len(front)}", ""] + head + [""]
    body += _digest_tables(scored, kept, front, MAG,
                           cfg["keep_ehull_meV"], rows, _A())
    (rdir / "digest.md").write_text("\n".join(body))

    print(f"merged {len(shards)} shard(s): {len(rows)} rows, {len(scored)} unique "
          f"scored ({n_xdup} cross-shard duplicates removed), {len(kept)} kept, "
          f"{len(front)} on the Pareto front")
    print(f"wrote {rdir}/digest.md  (this is what `propose` reads)")
    for r in front[:12]:
        print(f"  {r['file'][:44]:<46} E_hull {r['E_hull_meV']:7.1f} meV   "
              f"J_s_max {r['J_s_max_T']:.4f} T")


# --------------------------------------------------------------------------
def cmd_promote(a):
    """Pick k from the Pareto front and PREPARE a DFT campaign. Never submits."""
    loop = Path(a.loop)
    cfg = _cfg(loop)
    n = a.round or _current(loop)
    rdir = loop / f"round-{n:02d}"
    # --front promotes a CORRECTED shortlist instead of the screen's pareto.csv.
    # Not a convenience: pareto.csv ranks on the Fe-only J_s_max, and CHGNet puts
    # ~0 moment on the rare earth, so heavy-RE entries (Dy, Tb) sit HIGH on it and
    # then lose 30-35% once the antiparallel 4f moment is subtracted. Promoting
    # pareto.csv unchanged sends DFT exactly the structures the surrogate is most
    # wrong about, for the reason it is wrong about them. See correct_4f.py.
    fpath = Path(a.front) if a.front else (rdir / "pareto.csv")
    if not fpath.is_file():
        sys.exit(f"no front file at {fpath}")
    front = list(csv.DictReader(open(fpath)))
    if not front:
        sys.exit("empty Pareto front -- nothing to promote. Re-propose with a pivot.")
    if a.all:
        picks = front
    else:
        # Spread the k picks along the front rather than taking the k highest
        # J_s: the point of calibrating is to learn where the surrogate is wrong,
        # and that needs the cheap end of the front as well as the spectacular end.
        k = min(a.k, len(front))
        idx = [round(i * (len(front) - 1) / max(k - 1, 1)) for i in range(k)]
        picks = [front[i] for i in dict.fromkeys(idx)]
    dft = Path(a.outdir) if a.outdir else (rdir / "dft")
    (dft / "seeds").mkdir(parents=True, exist_ok=True)
    import shutil
    for p in picks:
        src = loop / "structures" / f"{p['fingerprint']}.vasp"
        if src.is_file():
            shutil.copy(src, dft / "seeds" / f"{p['file']}.vasp")
    (dft / "picks.json").write_text(json.dumps(picks, indent=2))

    # Write the campaign OURSELVES rather than printing `csp init`. `csp init`
    # builds a fresh campaign from a template and knows nothing about these
    # seeds, the mode, or the dedup rule -- so pointing at it means hand-editing
    # a campaign.yaml every round, which is both tedious and a place to get
    # `dedup` wrong. Everything the loop already knows goes in here.
    name = f"{Path(loop).name}-r{n:02d}-dft"
    natoms = max((int(float(p.get("natoms") or 0)) for p in picks), default=0)
    campaign = {
        "name": name,
        "machine": cfg.get("machine", "orion"),
        "workdir": f"/scratch/$USER/cspflow/{name}",
        "archive": f"/projects/mmi/Ridwan/cspflow_archive/{name}",
        "source": [{
            "mode": "structure_list", "name": "promoted",
            "structure_list": {
                "paths": ["inputs/seeds"],
                # MUST be True. relax:false inserts seeds already `screened` with
                # no mlip_e_per_atom, so reference_stage skips them, the hull
                # stays empty, filter selects nothing, and DFT NEVER RUNS --
                # `csp run` just prints `reference pending N` forever (found
                # 2026-09-15 on dft-campaign-01). Re-relaxing is also the more
                # correct choice: the MLIP energy in the database then belongs
                # to the same cell VASP receives. The loop's seeds are already
                # converged to fmax 0.05, so they barely move.
                "relax": True,
                # warn, never drop [C4]: two picks can share a composition and
                # differ only in WHICH site took the defect, and that difference
                # is the measurement. Dropping one deletes the answer.
                "dedup": "warn",
                "max_atoms": natoms + 8 if natoms else None,
            }}],
    }
    # THE ONE-SCALE RULE [E2] / D101. The store's settings file says it outright:
    # "Every campaign that consumes the computed store must carry it byte for
    # byte, or recipe_id differs and the 10,000 core-hours buy that campaign
    # nothing." A hand-written `dft: {recipe: magnets}` is NOT the same block --
    # csp doctor flags it, and the hull still builds, still looks right, and
    # ranks wrongly. So copy it rather than retype it.
    # Resolve the LIVE store's own settings first. Measured 2026-09-15: the
    # pinned path below is STALE -- the live store at $CSPFLOW_STORE has an extra
    # `dft.magnetism.table` of 8 elements that it does not, and because recipe_id
    # hashes the magnetism block WHOLESALE that single added key moves the recipe
    # (6e44a4bf351d98a9 vs 162acf1a59ace5f1). csp doctor caught it as a scale
    # mismatch. ADDING A KEY IS AS BREAKING AS CHANGING A VALUE, and none of the
    # 8 elements (As, Sb, Bi, Au, Ru, In, Se, Te) even occurs in these seeds.
    # So read the store that will actually answer, never a copy of it.
    _cands = []
    if os.environ.get("CSPFLOW_STORE_SETTINGS"):
        _cands.append(Path(os.environ["CSPFLOW_STORE_SETTINGS"]))
    if os.environ.get("CSPFLOW_STORE"):
        _cands.append(Path(os.environ["CSPFLOW_STORE"]) / "settings.yaml")
    _cands += [Path("/scratch/oridwan/mp-reference/settings.yaml"),
               Path("/projects/mmi/Ridwan/cspflow-reference/store-settings.yaml")]
    store_settings = next((c for c in _cands if c.is_file()), _cands[-1])
    if store_settings.is_file():
        campaign["dft"] = yaml.safe_load(store_settings.read_text())["dft"]
        scale_note = f"# dft: block copied verbatim from {store_settings} [E2]\n"
    else:
        campaign["dft"] = {"recipe": "magnets"}
        scale_note = ("# !! could not read the store's settings, so `dft:` is a BARE\n"
                      "# !! recipe name and will NOT match the reference scale. Run\n"
                      "# !! csp doctor and fix this before trusting any hull number.\n")
    campaign["source"][0]["structure_list"] = {
        k: v for k, v in campaign["source"][0]["structure_list"].items() if v is not None}
    (dft / "inputs").mkdir(exist_ok=True)
    if (dft / "inputs" / "seeds").exists():
        shutil.rmtree(dft / "inputs" / "seeds")
    shutil.move(str(dft / "seeds"), str(dft / "inputs" / "seeds"))
    corrected_rank = bool(picks[0].get("J_s_corrected_T"))
    how = ("the WHOLE front" if a.all else
           "spread ALONG the front, not the k highest J_s")
    (dft / "campaign.yaml").write_text(
        "# Written by orchestrate.py promote -- the round's DFT calibration set.\n"
        f"# {len(picks)} structures: {how}.\n"
        f"# Ranked on {'4f-CORRECTED J_s' if corrected_rank else 'J_s_max (Fe-only)'}"
        f"{' (see scripts/loop/correct_4f.py)' if corrected_rank else ''}.\n"
        "# The point of calibrating is to find where the surrogate is wrong, which\n"
        "# needs the cheap end of the front as well as the spectacular end.\n"
        "#\n# CHECK BEFORE RUNNING: recipe, workdir, archive, and whether this\n"
        "# campaign's DFT settings match the reference store's [E2].\n\n"
        + yaml.safe_dump(campaign, sort_keys=False))

    # VERIFY WHAT WE JUST WROTE. The dft: block is copied from the store so the
    # recipe matches, but nothing checked that it still did after the file was
    # written -- and a single flipped boolean in `rare_earth` (reconstruct_ms)
    # moves recipe_id and puts candidates and reference on different scales,
    # which is D101 all over again. The hull would still build and still rank
    # wrongly. Cheap to check here rather than hope someone runs csp doctor.
    try:
        import yaml as _y
        _written = _y.safe_load((dft / "campaign.yaml").read_text()).get("dft", {})
        _store = _y.safe_load(store_settings.read_text())["dft"]
        _hashed = ("potcar", "rare_earth", "magnetism", "ldau", "nbands",
                   "incar_overrides", "recipe")
        _bad = [k for k in _hashed
                if k in _store and _written.get(k) != _store.get(k)]
        if _bad:
            print(f"\n!! [E2] the dft: block does NOT match the store for {_bad}.")
            print(f"   store   {store_settings}")
            print("   recipe_id hashes these keys, so candidate and reference")
            print("   energies would land on DIFFERENT scales. Fix before running.")
        else:
            print(f"[E2] dft: block matches {store_settings} on every hashed key")
    except Exception as _exc:                                  # pragma: no cover
        print(f"!! could not verify the dft: block against the store: {_exc}")

    print(f"promoted {len(picks)} of {len(front)} front points to DFT "
          f"(from {fpath.name}):")
    for p in picks:
        js = p.get("J_s_corrected_T") or p.get("J_s_max_T")
        lab = "J_s(4f-corr)" if p.get("J_s_corrected_T") else "J_s_max     "
        print(f"  {p['file']:<44} E_hull {float(p['E_hull_meV']):7.1f} meV   "
              f"{lab} {float(js):.4f} T")
    print(f"\nwrote {dft}/campaign.yaml and {dft}/inputs/seeds/")
    print("\nNEXT, and this is yours to run -- the loop does not submit compute:")
    print(f"  csp doctor -c {dft}/campaign.yaml")
    print(f"  csp run    -c {dft}/campaign.yaml")
    print("\nthen, once those land:")
    print(f"  python {HERE/'orchestrate.py'} collect   --loop {loop} --round {n}")
    print(f"  python {HERE/'orchestrate.py'} calibrate --loop {loop} --round {n}")


def cmd_collect(a):
    """Read the finished VASP runs into results.csv, so `calibrate` has input.

    Walks for OUTCARs rather than reconstructing cspflow's directory layout,
    which differs between the combined and flat DFT modes. A structure is
    matched to its pick by its name appearing in the path, and the STATIC step
    wins over the relax step -- the magnetisation to trust is the one from the
    final self-consistent run, not from a geometry step.
    """
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from cspflow.dft.vasp.parse import read_job_directory
    from pymatgen.core import Structure

    loop = Path(a.loop).resolve()
    n = a.round or _current(loop)
    dft = loop / f"round-{n:02d}" / "dft"
    picks = json.loads((dft / "picks.json").read_text())
    roots = [Path(a.workdir)] if a.workdir else []
    if not roots:
        cy = yaml.safe_load((dft / "campaign.yaml").read_text())
        roots = [Path(os.path.expandvars(cy["workdir"])),
                 Path(os.path.expandvars(cy["archive"]))]
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for oc in root.rglob("OUTCAR"):
            d = oc.parent
            hit = next((p["file"] for p in picks if p["file"] in str(d)), None)
            if not hit:
                continue
            static = "static" in d.name.lower()
            if hit in found and not static:
                continue          # keep the static one
            out = read_job_directory(d)
            if out.energy is None or out.n_atoms is None:
                continue
            vol, mag_sum = None, None
            st = None
            for g in ("CONTCAR", "POSCAR"):
                if (d / g).is_file():
                    try:
                        st = Structure.from_file(d / g)
                        vol = st.volume
                        break
                    except Exception:
                        pass
            # Per-site moments, so the MAGNETIC SUBLATTICE sum can be compared
            # like with like. The cell total from OSZICAR is not the right
            # quantity: the store's recipe is f_treatment: frozen with
            # reconstruct_ms: true, so the rare-earth part of a reported total is
            # a Hund's-rule assumption, not a measurement [D1]. CHGNet's
            # prediction is for the sublattice; calibrate against that.
            try:
                from pymatgen.io.vasp.outputs import Outcar
                per_ion = [m.get("tot", 0.0) for m in (Outcar(d / "OUTCAR").magnetization or [])]
                if st is not None and len(per_ion) == len(st):
                    mag_sum = sum(m for m, t in zip(per_ion, st)
                                  if str(t.specie) == a.magnetic_element)
            except Exception:
                pass
            found[hit] = {"file": hit, "e_per_atom": round(out.e_per_atom, 6),
                          "m_total_muB": out.magnetisation,
                          f"m_{a.magnetic_element}_sum_muB":
                              round(mag_sum, 4) if mag_sum is not None else "",
                          "volume_A3": round(vol, 3) if vol else "",
                          "n_atoms": out.n_atoms, "step": d.name,
                          "state": out.state}
    if not found:
        sys.exit(f"no finished OUTCARs matched the {len(picks)} promoted names under "
                 f"{', '.join(str(r) for r in roots)}.\n"
                 f"Pass --workdir if the campaign ran somewhere else.")
    rows = [found[p["file"]] for p in picks if p["file"] in found]
    out = dft / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    missing = [p["file"] for p in picks if p["file"] not in found]
    print(f"collected {len(rows)} of {len(picks)} promoted structures -> {out}")
    for r in rows:
        print(f"  {r['file'][:44]:<46} {r['state']:<8} {r['e_per_atom']:>10.4f} eV/at  "
              f"m {r['m_total_muB']}")
    if missing:
        print(f"\nSTILL MISSING ({len(missing)}): {', '.join(missing)}")
        print("  -- calibrate will run on what is here, but a drift measured on a")
        print("     partial set is weaker evidence. Say so if you report it.")


# --------------------------------------------------------------------------
def cmd_calibrate(a):
    """Measure the surrogate against DFT. The loop's only honesty check."""
    loop = Path(a.loop)
    n = a.round or _current(loop)
    rdir = loop / f"round-{n:02d}"
    picks = json.loads((rdir / "dft" / "picks.json").read_text())
    dftcsv = rdir / "dft" / "results.csv"
    if not dftcsv.is_file():
        sys.exit(f"{dftcsv} not found. Expected columns: file,e_per_atom,m_total_muB,volume_A3")
    got = {r["file"]: r for r in csv.DictReader(open(dftcsv))}
    C = 4.0e-7 * 3.141592653589793 * 9.2740100783e-24 / 1.0e-30
    L = [f"# Round {n} calibration", "",
         "| file | J_s_max (CHGNet) | J_s (DFT) | error | E_hull (MLIP) |",
         "|---|---|---|---|---|"]
    errs = []
    MAG = a.magnetic_element
    sub_key, sur_key = f"m_{MAG}_sum_muB", f"m_{MAG}_sum_muB"
    # BOTH sides must carry the sublattice sum, or the comparison is not the one
    # the header claims. Deciding this from the DFT side alone printed
    # "comparing the Fe-sublattice moment" over a row whose surrogate value was
    # still a cell total -- a caption that misdescribes its own table.
    using_sublattice = (any(g.get(sub_key) for g in got.values())
                        and all(p.get(sur_key) for p in picks))
    L.insert(1, ("_Comparing the " + MAG + "-sublattice moment on both sides_ -- a "
                 "computed number in each case." if using_sublattice else
                 "_Comparing CELL TOTALS: the " + MAG + "-sublattice sum is missing on "
                 "one side (older pareto.csv, or per-site moments unreadable). With "
                 "f_treatment: frozen the DFT total carries a reconstructed 4f term "
                 "[D1], so this comparison is weaker than it looks -- re-run the round "
                 "to get m_" + MAG + "_sum_muB on the surrogate side._"))
    for p in picks:
        g = got.get(p["file"])
        if not g:
            continue
        if using_sublattice and g.get(sub_key) and p.get(sur_key):
            m_dft, m_sur = float(g[sub_key]), float(p[sur_key])
        else:
            m_dft, m_sur = float(g["m_total_muB"]), None
        js_dft = C * m_dft / float(g["volume_A3"])
        js_sur = (C * m_sur / float(g["volume_A3"])) if m_sur is not None \
            else float(p["J_s_max_T"])
        e = js_sur - js_dft
        errs.append(e / js_dft * 100)
        L += [f"| {p['file'][:38]} | {js_sur:.4f} | {js_dft:.4f} | "
              f"{e:+.4f} ({e/js_dft*100:+.1f}%) | {float(p['E_hull_meV']):.1f} |"]
    if errs:
        import statistics as st
        L += ["", f"- mean signed drift: {st.mean(errs):+.2f}%",
              f"- spread: {(max(errs)-min(errs)):.2f} percentage points", "",
              "A drift that GROWS round on round is the loop learning the "
              "surrogate's error rather than the physics [G9]. If it does, stop "
              "and re-anchor: the Pareto front is being chosen by CHGNet's bias.",
              "", "A LARGE POSITIVE drift on one structure is the ferrimagnetic "
              "signature -- CHGNet summed magnitudes where DFT found cancellation "
              "[D3]. Check that structure's per-site signs before trusting it."]
    out = rdir / "calibration.md"
    out.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {out}")


def cmd_status(a):
    loop = Path(a.loop)
    cfg = _cfg(loop)
    print(f"loop: {loop}\nparent: {cfg['parent']}\n")
    arch = loop / "archive.csv"
    total = sum(1 for _ in csv.DictReader(open(arch))) if arch.is_file() else 0
    print(f"archive: {total} structures scored across all rounds\n")
    for d in _round_dirs(loop):
        bits = []
        for name in ("plan.yaml", "scored.csv", "pareto.csv", "digest.md",
                     "dft/picks.json", "calibration.md"):
            bits.append(("+" if (d / name).is_file() else "-") + name.split("/")[0])
        print(f"  {d.name}: {' '.join(bits)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init");  p.set_defaults(fn=cmd_init)
    p.add_argument("--loop", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--intuition", default=None)
    p.add_argument("--elements", nargs="+", default=None)
    p.add_argument("--magnetic-element", default="Fe")
    p.add_argument("--mask-elements", nargs="*", default=[])
    p.add_argument("--keep-ehull", type=float, default=200.0,
                   help="absolute MatterSim E_hull ceiling, meV/atom")
    p.add_argument("--seeds-per-round", type=int, default=60)
    p.add_argument("--dft-per-round", type=int, default=5)
    p.add_argument("--machine", default="orion")

    for name, fn in (("propose", cmd_propose), ("generate", cmd_generate),
                     ("screen", cmd_screen), ("merge", cmd_merge),
                     ("promote", cmd_promote), ("collect", cmd_collect),
                     ("calibrate", cmd_calibrate), ("status", cmd_status)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--loop", required=True)
        p.add_argument("--round", type=int, default=None)
        if name == "calibrate":
            p.add_argument("--magnetic-element", default="Fe")
        if name == "promote":
            p.add_argument("-k", type=int, default=5)
            p.add_argument("--front", default=None,
                           help="CSV to promote instead of the round's pareto.csv, "
                                "e.g. a 4f-corrected shortlist")
            p.add_argument("--all", action="store_true",
                           help="promote every row of the front, not k spread picks")
            p.add_argument("--outdir", default=None,
                           help="where to write the campaign (default round-NN/dft)")
        if name == "collect":
            p.add_argument("--magnetic-element", default="Fe")
            p.add_argument("--workdir", default=None,
                           help="where the DFT ran, if not the campaign's own workdir")
        if name == "screen":
            p.add_argument("--chunk", type=int, default=40)
            p.add_argument("--local", action="store_true",
                           help="run here instead of submitting. Debug only.")
        if name == "propose":
            p.add_argument("--dry-run", action="store_true")

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
