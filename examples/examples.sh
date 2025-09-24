#!/bin/bash

#SBATCH -J strain
#SBATCH -o example_script-%j.out
#SBATCH -p gpu2
#SBATCH --gpus-per-node 1
#SBATCH --mem 10GB

# script should be run from the StrainRelief root directory

source ~/StrainRelief/.venv/bin/activate  # with uv
# mamba activate strain  # with conda/mamba

# You must choose a minimisation and energy evaluation method from "mmff94",
# "mmff94s", "mace" or "fairchem". The calculator works best when the same
# force field is used for both methods. If this is the case, "energy_eval"
# does not need to be specified.

# This is the simplest and fastest implementation of StrainRelief using MMFF94s
# and a minimial example dataset.
strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.parquet_path=../data/example_ligboundconf_output.parquet \
    minimisation@global_min=mmff94s \
    minimisation@local_min=mmff94s

# This script demonstrates using different force fields for minimisation
# (MMFF94s) and energy evaluations (MACE).
strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.parquet_path=../data/example_ligboundconf_output.parquet \
    minimisation@global_min=mmff94s \
    minimisation@local_min=mmff94s \
    energy_eval=mace \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model

# This is the script as used for most calculations in the StrainRelief paper.
# MACE is used for minimisation (and energy evalutions implicitly). A looser
# convergence criteria is used for local minimisation. Note: a gpu is required
# by default to run calculations with MACE.
strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.parquet_path=../data/example_ligboundconf_output.parquet \
    minimisation@global_min=mace \
    minimisation@local_min=mace \
    local_min.fmax=0.50 \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    hydra.verbose=true
