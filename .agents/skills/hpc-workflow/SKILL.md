---
name: hpc-workflow
description: Configure and operate this project's SSH and Slurm jobs on the UMT cluster, including persistent submission records, recovery and result retrieval.
---

Read `configs/cluster.json`, `configs/ssh_config` and the current project state. Resolve null fields before preparing runnable production submissions.

Verify the server key through a trusted channel before adding it to local known_hosts. A keyscan is a candidate key, not independent verification. Keep authentication machine-local. Never expose private keys or disable host checking to get past connection errors.

Inspect partitions, account/QOS limits, node features, GPU models, modules, VASP/MPI installation and writable storage. Use lightweight login-node commands; execute compute work through Slurm. Reported installed hardware is not a promised allocation. Check K40m compatibility independently from A100, including compute capability, framework binaries and driver/runtime constraints.

Prepare a unique immutable run directory containing configuration, code revision, input hashes, environment identity and requested resources. Persist a submission intent before invoking sbatch. Capture the returned job ID immediately. If the connection drops during submission, look for the matching run in queue/accounting records before any retry; ambiguous state requires reconciliation.

Track queued/running/completed/failed/cancelled and an explicit unknown state. Scheduler completion is not scientific success: check exit code, expected outputs, VASP electronic/ionic convergence and parser validation. Bound retries and record every attempt; do not retry a deterministic scientific/input failure unchanged.

Retrieve outputs with checksums and confirm durability before marking a run collected. Avoid destructive sync. Record the exact remote directory and cluster identity with job IDs. Long-running jobs must remain traceable after the local chat or laptop stops.
