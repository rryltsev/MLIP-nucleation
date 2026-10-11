# MLIP nucleation research

Read `README.md`, `project/state.json`, and `project/brief.txt` before continuing work. The brief defines the scientific objective; hypotheses are not results. User-facing discussion is in Russian. Manuscripts and reports use LaTeX; code and configuration use English.

## Research invariants
- Study heterogeneous nucleation mediated by interfacial liquid order. Alloy/nucleant selection remains open; Al-based candidates are not a committed choice.
- Preserve DFT labels separately from teacher predictions. Record energy, force and stress units, stress sign/order, electronic settings and label provenance.
- Split datasets by parent structure/trajectory/system before fitting. Never select models or tune sampling using the held-out DFT test set.
- Validate teacher and student against DFT and relevant physical observables. Record failed runs and negative evidence.
- Every published number and figure must trace to input hashes, code commit, model hash, analysis settings and source runs. Never invent results or citations.

## Execution and persistence
- VASP and production simulations run on Slurm compute nodes. Login-node work is limited to lightweight inspection and transfers.
- Before submission, resolve partition, account, resources, walltime, software and user-authorized allocation budget. Do not infer available capacity from total installed hardware.
- Persist a run identifier and input hash before submission; record scheduler job ID immediately. If submission outcome is uncertain, reconcile scheduler state before retrying.
- Never disable SSH host verification. Credentials, private keys, tokens and licensed VASP pseudopotential files stay outside Git.
- Commit coherent code/configuration/result changes; push to the configured GitHub remote when authentication and Git identity are available. Do not fabricate commit author identity.
- Store large artifacts through an explicitly chosen backend with checksums; manifests alone are not backups. No destructive synchronization.
- Update `project/state.json` after meaningful progress so another computer can resume. Do not commit mutable live scheduler databases.

## Project skills
Skills live in `.agents/skills/` for repository-scoped discovery. Read the matching `SKILL.md` when performing the corresponding task.
- `.agents/skills/mlip-study/SKILL.md`: study design and scientific continuation.
- `.agents/skills/hpc-workflow/SKILL.md`: SSH/Slurm execution and recovery.
- `.agents/skills/mlip-distillation/SKILL.md`: DFT to teacher to student workflow.

Use immutable run directories and separate working branches/checkouts for simultaneous work. This file does not request automatic agent delegation.
