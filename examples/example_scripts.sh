#!/bin/bash

#SBATCH -J strain
#SBATCH -o example_script-%j.out
#SBATCH -p gpu2
#SBATCH --gpus-per-node 1
#SBATCH --mem 10GB

# scripts should be run from the StrainRelief root directory

source ~/StrainRelief/.venv/bin/activate  # with uv
# mamba activate strain  # with conda/mamba

# You must choose a minimisation and energy evaluation method from "mmff94",
# "mmff94s", "mace" or "fairchem". The calculator works best when the same
# force field is used for both methods. If this is the case, "energy_eval"
# does not need to be specified.

# This is the simplest and fastest implementation of StrainRelief using MMFF94s
# and a minimial example dataset.
strain-relief \
    io.input.parquet_path=./data/example_ligboundconf_input.parquet \
    io.output.parquet_path=./data/example_ligboundconf_output.parquet \
    optimiser@local_optimiser=bfgs \
    optimiser@global_optimiser=bfgs \
    calculator=mmff94s \
    device=cpu

# This script demonstrates using different force fields for minimisation
# (MMFF94s) and energy evaluations (MACE).
strain-relief \
    io.input.parquet_path=./data/example_ligboundconf_input.parquet \
    io.output.parquet_path=./data/example_ligboundconf_output.parquet \
    optimiser@local_optimiser=bfgs \
    optimiser@global_optimiser=bfgs \
    calculator=mmff94s \
    energy_evaluation.calculator=mace \
    energy_evaluation.calculator.model_paths=./tests/models/MACE.model \
    device=cpu

# This is the script as used for most calculations in the StrainRelief paper.
# MACE is used for minimisation (and energy evalutions implicitly). A looser
# convergence criteria is used for local minimisation.
strain-relief \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.parquet_path=../data/example_ligboundconf_output.parquet \
    optimiser@local_optimiser=bfgs \
    optimiser@global_optimiser=bfgs \
    calculator=mace \
    calculator.model_paths=./tests/models/MACE.model \
    hydra.verbose=true \
    device=cuda
