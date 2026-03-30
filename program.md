# Program

## User request

Run two new baseline trainings on CINECA with the SLURM launcher, capping training at 20 minutes and comparing `cpus-per-task=16` versus `cpus-per-task=32`.

## Steps

1. Ensure the baseline `sp1024` dataset is present with the expected train shards.
2. Update `.env` so the default single-GPU wallclock cap is 20 minutes.
3. Submit one baseline job with `cpus-per-task=16` and matching CPU thread env vars.
4. Submit one baseline job with `cpus-per-task=32` and matching CPU thread env vars.
5. Record the submitted job ids and verify their SLURM states.
