---
name: mlip-distillation
description: Build and validate the project's VASP-labelled dataset, fine-tuned foundation teacher and distilled student for interfacial liquid and nucleation simulations.
---

Resolve the selected chemistry, thermodynamic domain, competing phases and target observables first. Choose actual model/backend versions from current primary documentation and available hardware; no foundation checkpoint or student architecture is fixed by this skill.

Cover bulk liquid/solid, interfaces, relevant surface terminations, segregation, strained structures and physically relevant short separations. Assign stable configuration and parent trajectory IDs. Preserve raw VASP outputs or durable artifact references, settings, pseudopotential identifiers/hashes, magnetism, smearing and convergence status. Never upload licensed POTCAR content.

Normalize units and stress/virial conventions explicitly; distinguish total energies from per-atom errors. Reject unconverged labels. Document atomic reference energy treatment; do not mix incompatible DFT reference levels without a justified transformation.

Freeze grouped train/validation/test membership before training. Prevent related trajectory frames or parent structures leaking across splits. Separate task fine-tuning data from replay data and check overlap with held-out tests where provenance permits.

Evaluate the starting FM and fine-tuned teacher on independent DFT energies, forces, stresses and relevant physical observables. Specify acceptable errors for the chosen observable. Do not generate a distillation corpus from an unvalidated teacher and treat it as DFT truth.

Freeze the teacher checkpoint and hash. Store pseudo-label provenance separately from DFT labels and record sampling conditions and any weighting of mixed sources. Train a student selected for measured throughput and target accuracy, not parameter count alone.

Evaluate student versus both teacher and independent DFT. Check interfacial segregation, liquid order, phase energetics and MD stability as relevant. Benchmark exported/deployed models against their training implementation on identical structures, and measure speed on the intended system size/hardware.

Use active learning only with explicit selection criteria and DFT budgets. Ensemble disagreement is not automatically calibrated uncertainty. Version every acquisition round without using the frozen test set for selection. Publish model applicability limits and failure cases with the accepted checkpoint.
