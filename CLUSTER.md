# CLUSTER.md — Leonardo HPC Quick Reference

This guide covers the essentials for running jobs on **Leonardo** at CINECA.
Full docs: <https://docs.hpc.cineca.it>

---

## 1. Connection

```bash
ssh <username>@login.leonardo.cineca.it
```

Login nodes: `login01-ext` through `login04-ext`. Use `login01-ext` specifically when setting up the squid proxy (see [Internet on Compute Nodes](#7-internet-on-compute-nodes)).

---

## 2. Filesystems

| Variable | Path example | Quota | Backup | Purging | Notes |
|----------|-------------|-------|--------|---------|-------|
| `$HOME` | `/leonardo/home/userexternal/<user>` | 50 GB | Yes | No | Personal, small configs/code |
| `$WORK` | `/leonardo_work/<account>` | 1 TB | No | No | **Project-shared**, persistent, local |
| `$FAST` | `/leonardo_scratch/fast/<account>` | 1 TB | No | No | **Project-shared**, persistent, local, fast I/O |
| `$SCRATCH` | `/leonardo_scratch/large/userexternal/<user>` | — | No | **Yes (auto-purge)** | Per-user, large temporary storage |

### Conventions for this project

- Each collaborator creates a **personal directory** under `$WORK` and `$FAST`:
  ```bash
  mkdir -p $WORK/$USER
  mkdir -p $FAST/$USER
  ```
- Personal directories are **protected by default** (only the owner can read/write).
- To allow project collaborators access, use `chmod`:
  ```bash
  chmod 750 $WORK/$USER   # group can read+execute
  ```
- `$WORK` and `$FAST` are owned by the PI; all collaborators have read/write access to the top-level directory.
- **Neither `$WORK` nor `$FAST` are backed up** — keep important code in git.
- Use `$SCRATCH` for large throwaway data (model checkpoints, datasets). It is **auto-purged** after ~30 days of inactivity. Since `$WORK` has a **1 TB limit** shared across the project, really heavy data (large datasets, full checkpoint histories, etc.) should go in `$SCRATCH` to avoid filling up `$WORK`.
- Use `$HOME` for dotfiles, ssh keys, and small configs only.

---

## 3. Partitions

| Partition | GPUs | CPUs/node | RAM/node | Max walltime | Use case |
|-----------|------|-----------|----------|-------------|----------|
| `boost_usr_prod` | 4x A100 (64 GB) | 32 | 512 GB | 24h | **Main GPU partition** |
| `dcgp_usr_prod` | — (CPU only) | 112 | 512 GB | 24h | CPU-heavy workloads |
| `lrd_all_serial` | — | 128 | 512 GB | 4h | Login-node serial jobs |
| `lrd_all_viz` | 2x Quadro | 128 | 512 GB | 12h | Visualization |

For our workloads, use **`boost_usr_prod`** with `--gres=gpu:1` (or more).

---

## 4. SLURM Cheat Sheet

### Submit a job

```bash
sbatch scripts/launch_my_job.sh
```

### SBATCH header template

```bash
#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_TVU/<user>/spectral-compression
#SBATCH --job-name=my-job
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU

set -euo pipefail

module load cuda/12.2

# Internet access (see section 7)
export http_proxy='http://login01:<YOUR_PORT>'
export https_proxy='http://login01:<YOUR_PORT>'

mkdir -p slurm results

srun uv run python scripts/my_script.py
```

### Useful commands

```bash
sbatch script.sh          # Submit a batch job
squeue -u $USER           # Check your jobs
scancel <job_id>          # Cancel a job
scancel -u $USER          # Cancel all your jobs
sinfo -o "%P %l %G %c %m" # Partition overview
sacct -j <job_id>         # Job accounting info after completion
srun --pty bash           # Interactive shell on compute node
```

### Inspect GPU usage inside a running job

If `nvitop` is installed, attach to the allocated node for a running job and open it directly:

```bash
srun --jobid 38586553 --overlap --pty bash -lc 'cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf && uv run nvitop'
```

Fallback with `nvidia-smi`:

```bash
srun --jobid <job_id> --overlap --pty bash -lc 'watch -n 1 nvidia-smi'
```

### Interactive GPU session

```bash
srun -p boost_usr_prod -A IscrC_TVU --gres=gpu:1 --mem=40G --time=2:00:00 --pty bash
```

### Request multiple GPUs

```bash
#SBATCH --gres=gpu:2      # 2 GPUs on one node
#SBATCH --ntasks-per-node=2
```

---

## 5. Module System

```bash
module avail              # List all available modules
module load cuda/12.2     # Load CUDA
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
module list               # Show loaded modules
module purge              # Unload everything
```

For deep learning workloads, load profiles first:
```bash
module load profile/deeplrn
module avail              # Now shows DL-specific modules
```

---

## 6. Python / uv

We use **[uv](https://docs.astral.sh/uv/)** for dependency management. `uv` is not bundled in the repository, so install it first if it is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync                   # Install deps from pyproject.toml
uv run python script.py   # Run with the project's venv
uv run pytest             # Run tests
```

In SLURM scripts, prefix commands with `srun uv run` so they execute on the compute node with the correct environment.

---

## 7. Internet on Compute Nodes

Leonardo compute nodes have **no internet access by default**. We work around this with a custom `squid` proxy running on a login node.

### Using the existing proxy (most collaborators)

Add these two lines to your SLURM scripts and/or `~/.bashrc`:

```bash
export http_proxy='http://login01:<YOUR_PORT>'
export https_proxy='http://login01:<YOUR_PORT>'
```

Ask the project lead for the port number to use, or set up your own (see below).

For `~/.bashrc`, gate it so it only applies on compute nodes:

```bash
if [[ $HOSTNAME == *"lrdn"* || $SLURMD_NODENAME == *"lrdn"* ]]; then
  export http_proxy='http://login01:<YOUR_PORT>'
  export https_proxy='http://login01:<YOUR_PORT>'
fi
```

### Setting up your own squid proxy

If you need your own proxy (e.g., your port conflicts or the existing one is down):

1. **SSH into `login01-ext`** (must be login01, not the generic login):
   ```bash
   ssh <user>@login01-ext.leonardo.cineca.it
   ```

2. **Download and extract squid**:
   ```bash
   wget https://github.com/squid-cache/squid/releases/download/SQUID_6_10/squid-6.10.tar.gz
   tar xzf squid-6.10.tar.gz && cd squid-6.10
   ```

3. **Patch `src/ipc/mem/Segment.cc`** — in the functions `open()`, `createFresh()`, and `unlink()`, replace the shared-memory names with a unique prefix (e.g., `/mysquid_<yourname>`) to avoid collisions with the system squid. The full patch is documented in the [Notion guide](https://www.notion.so/crisostomi/Enabling-internet-on-compute-nodes-1ca9a06bd524803c952ee39d6684b8e4).

4. **Compile and install**:
   ```bash
   mkdir install
   ./configure --prefix=$(pwd)/install
   make -j4 && make install
   ```

5. **Edit `install/etc/squid.conf`**:
   - `http_access deny all` → `http_access allow all`
   - `http_port 3128` → `http_port <YOUR_PORT>`
   - Add: `dns_v4_first on`
   - Add: `cache_log /path/to/squid-6.10/install/cache.log`

6. **Start it**: `cd install/sbin && ./squid`

7. **Verify**: `ss -ltpn | grep squid`

**Already taken ports**: 1821, 1909, 2302, 3128, 3129, 3133, 3140, 3141, 3142, 3169, 3420. Pick something else.

### When the proxy dies

Rarely happens, but if it does:
- SSH to `login01-ext` and restart: `cd ~/squid-6.10/install/sbin && ./squid`
- If `login01` itself is down, start squid on another login node and update the proxy URLs in your scripts accordingly.

### Debugging

```bash
# On compute node, check if proxy env vars are set:
env | grep proxy

# Test connectivity:
curl https://example.com
```

---

## 8. Tips

- **Always `mkdir -p slurm`** before submitting jobs — SLURM will fail silently if the output directory doesn't exist.
- **Use `--resume`** flags in long-running scripts so you can restart from the last checkpoint if a job times out.
- **Check GPU health** at the start of your job:
  ```bash
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  ```
- **Max walltime is 24h** on `boost_usr_prod`. For longer runs, implement checkpointing and resubmit.
- **Disk I/O**: prefer `$FAST` for data that needs high throughput during training. Use `$WORK` for persistent results.
- **Keep `$HOME` light** — it has a 50 GB quota and is the only filesystem that is backed up.
