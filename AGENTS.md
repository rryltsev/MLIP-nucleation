# MLIP nucleation research

Read `README.md`, `project/state.json`, and `project/brief.txt` before continuing work. The brief defines the scientific objective; hypotheses are not results. User-facing discussion is in Russian. Manuscripts and reports use LaTeX; code and configuration use English.

## Research invariants
- Study heterogeneous nucleation mediated by interfacial liquid order. Alloy/nucleant selection remains open; Al-based candidates are not a committed choice.
- Preserve DFT labels separately from teacher predictions. Record energy, force and stress units, stress sign/order, electronic settings and label provenance.
- Split datasets by parent structure/trajectory/system before fitting. Never select models or tune sampling using the held-out DFT test set.
- Validate teacher and student against DFT and relevant physical observables. Record failed runs and negative evidence.
- Current architecture priorities are DPA4, DPA4c and GRACE, with extensions allowed. Use `configs/model_backends.json`; architecture availability is not checkpoint readiness. A100 synthetic DPA4/DPA4c training is verified in `lammps_deepmd_dpa4`; K40m verification covers only legacy `se_e2_a`. Keep GRACE separate from existing DeepMD environments because of its TensorFlow version constraint.
- Every published number and figure must trace to input hashes, code commit, model hash, analysis settings and source runs. Never invent results or citations.

## Execution and persistence
- VASP and production simulations run on Slurm compute nodes. Login-node work is limited to lightweight inspection and transfers.
- The main home filesystem was read-only on compute node apollo20. Place build trees, simulation working directories and Slurm logs on a compute-writable filesystem; write access to `/home6/u3129` was verified through Slurm. Keep durable copies of accepted software/results in main home or the chosen archive.
- For compute-node environment installation, isolate installer HOME and caches on writable storage. Overriding only `CONDA_PKGS_DIRS` is insufficient: Conda plugins can still try writing to the read-only main home. Preserve the existing environments.
- Compute-node Lmod does not expose the login node's `intelmpi/intel64_2025` module. Use the pinned environment setup in `scripts/hpc/vasp-env.sh` for this VASP build.
- Before submission, resolve partition, account, resources, walltime, software and user-authorized allocation budget. Do not infer available capacity from total installed hardware.
- Persist a run identifier and input hash before submission; record scheduler job ID immediately. If submission outcome is uncertain, reconcile scheduler state before retrying.
- Never disable SSH host verification. Credentials, private keys, tokens and licensed VASP pseudopotential files stay outside Git.
- For VASP inputs use the private PAW library registered in `configs/vasp/paw-64.json` and the hash-checking assembly helper documented in README. Choose PBE or LDA and exact variants explicitly in POSCAR species order; preserve `potcar-provenance.json` in each run's records. Do not mix potential families/variants silently or substitute bundled test potentials.
- Commit coherent code/configuration/result changes; push to the configured GitHub remote when authentication and Git identity are available. Do not fabricate commit author identity.
- Store large artifacts through an explicitly chosen backend with checksums; manifests alone are not backups. No destructive synchronization.
- Update `project/state.json` after meaningful progress so another computer can resume. Do not commit mutable live scheduler databases.

## Project skills
Skills live in `.agents/skills/` for repository-scoped discovery. Read the matching `SKILL.md` when performing the corresponding task.
- `.agents/skills/mlip-study/SKILL.md`: study design and scientific continuation.
- `.agents/skills/hpc-workflow/SKILL.md`: SSH/Slurm execution and recovery.
- `.agents/skills/mlip-distillation/SKILL.md`: DFT to teacher to student workflow.

Use immutable run directories and separate working branches/checkouts for simultaneous work. This file does not request automatic agent delegation.
