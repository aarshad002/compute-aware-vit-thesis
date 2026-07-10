# logs/

Run logs only — nothing in here is an input to any script; all files are outputs
kept for the record.

| Folder | Contents |
|--------|----------|
| `slurm/` | SLURM job logs, one pair per job: `<jobname>_<jobid>.out` (stdout) and `.err` (stderr). Also the default-named `slurm-<jobid>.out` files. |
| `run_transcripts/` | Raw stdout transcripts captured from individual experiment runs (e.g. `cascade_imagenet.log`, `l3_cascade.log`, `subdense_cascade.log`). The large `*.log` files are gitignored. |

New SLURM runs write to `logs/` (see `#SBATCH --output=logs/%x_%j.out` in
`scripts/run_hpc.sh` / `scripts/run_script.sh`); move finished job logs into
`slurm/` to keep the top level tidy.
