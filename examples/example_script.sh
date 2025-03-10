#!/bin/bash

#SBATCH -J strain
#SBATCH -o example_script-%j.out
#SBATCH -p gpu2
#SBATCH --gpus-per-node 1
#SBATCH --mem 10GB

# script should be run from presient/strain_relief directory

strain-relief \
    seed=123 \
    experiment=mace \
    conformers.numConfs=5 \
    io.input.parquet_path=../data/example_ligboundconf_input.parquet \
    io.output.output_file=../data/example_ligboundconf_output.parquet \
    hydra.verbose=true
