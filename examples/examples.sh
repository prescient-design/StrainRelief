#!/bin/bash

#SBATCH -J strain
#SBATCH -o example_script-%j.out
#SBATCH -p gpu2
#SBATCH --gpus-per-node 1
#SBATCH --mem 10GB

# script should be run from presient/strain_relief directory

source ~/prescient/strain_relief/.venv/bin/activate  # with uv
# mamba activate strain  # with conda/mamba

strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.output_file=../data/example_ligboundconf_output.parquet \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s

strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.output_file=../data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s \
    energy_eval=mace \
    energy_eval.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model

strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.output_file=../data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@local_min=mace \
    local_min.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@global_min=mace \
    global_min.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    local_min.fmax=0.50 \
    hydra.verbose=true
