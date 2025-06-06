# StrainRelief 💊
StrainRelief calculates the ligand strain of uncharged docked poses and has a suite of different force fields with which to do this. This includes a MACE neural network potential trained on SPICE2.

The pre-print can be found [here](https://arxiv.org/abs/2503.13352) and all relevant datasets [here](https://huggingface.co/datasets/erwallace/LigBoundConf2.0).

![Strain Relief Logo](assets/strain_relief_logo.png)

## Installation

From the root directory, run the following commands to install the package and its dependencies in editable mode:

(`mace-torch==0.3.x` requires `e3nn==0.4.4` (only for training, not inference). `fairchem-core` requires `e3nn>=0.5`. So until `mace-torch==0.4` is released we will have to do this finicky way of installing [issue](https://github.com/ACEsuit/mace/issues/555))

```bash
mamba env create -f env.yml
mamba activate strain
pip install -e .

pip install --force-reinstall e3nn==0.5 fairchem-core

pre-commit install
```

### Installation with uv
Install `uv` and from the root directory run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```


## The Protocol

The protocol used in StrainRelief is designed to be simple, fast and model agnostic - all that is needed to apply a new force field is to write an ASE calculator wrapper. Additionally you can use any MACE model, such as these from the [MACE-OFF23](https://github.com/ACEsuit/mace-off/tree/main/mace_off23) repository.

![Strain Relief Protocol](assets/strain_relief_protocol.png)

The protocol consists of 5 steps:

1. Minimise the docked pose with a loose convergence criteria to give a local minimum.
2. Generate 20 conformers from the docked ligand pose.
3. Minimise the generated conformers (and the original docked pose) with a stricter convergence criteria.
4. Evaluate the energy of all conformers and choose the lowest energy as an approximation of the global minimum.
5. Calculate `E(ligand strain) = E(local minimum) - E(global minimum)` and apply threshold.

**N.B.** energies returned are in kcal/mol.

## Usage
Choose a minimisation and energy evalation force field from `mmff94`, `mmff94s`, `mace`.

The calculator works best when the same force field is used for both methods. If this is the case, `energy_eval` does not need to be specified.

See the example scripts in [examples](./examples/examples.sh) along with a [tutorial](./examples/tutorial.ipynb) to explain StrainRelief's output and some handy helper functions.

This is the simplest and fastest implementation of StrainRelief using MMFF94s and a minimial example dataset.
```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s \
    local_min.fmax=0.50
```

This script demonstrates using different force fields for minimisation (MMFF94s) and energy evaluations (MACE).
```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=models/MACE_SPICE2_NEUTRAL.model \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s \
    energy_eval=mace \
    energy_eval.model_paths=models/MACE_SPICE2_NEUTRAL.model
```

This is the script as used for most calculations in the StrainRelief paper. MACE is used for minimisation (and energy evalutions implicitly). A looser convergence criteria is used for local minimisation. Note: a gpu is required by default to run calculations with MACE.
```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=models/MACE_SPICE2_NEUTRAL.model \
    minimisation@local_min=mace \
    local_min.model_paths=models/MACE_SPICE2_NEUTRAL.model \
    minimisation@global_min=mace \
    global_min.model_paths=models/MACE_SPICE2_NEUTRAL.model \
    local_min.fmax=0.50 \
    hydra.verbose=true
```

#### RDKit kwargs
The following dictionaries are passed directly to the function of that name.
- `conformers` (`EmbedMultipleConfs`)
- `minimisation.MMFFGetMoleculeProperties`
- `minimisation.MMFFGetMoleculeForceField`
- `energy_eval.MMFFGetMoleculeProperties`
- `energy_eval.MMFFGetMoleculeForceField`

The hydra config is set up to allow additional kwargs to be passed to these functions e.g. `+minimisation.MMFFGetMoleculeProperties.mmffVerbosity=1`.

**Common kwargs**
- `threshold` (set by default to 16.1 kcal/mol - calibrated using [LigBoundConf 2.0](https://huggingface.co/datasets/erwallace/LigBoundConf2.0))
- `conformers.numConfs`
- `global_min.maxIters`/`local_min.maxIters`
- `global_min.fmax`/`local_min.maxIters`
- `hydra.verbose`
- `seed`

#### Input Data
`strain-relief` accepts pd.DataFrames with RDKit molecules stored as `bytes` strings (using `mol.ToBinary()`)

### Logging

Logging is set to the `INFO` level by default which logs only aggregate information. `hydra.verbose=true` can be used to activate `DEBUG` level logging which includes information for every molecule and conformer.

## Unit Tests
- `pytest tests/` - runs all tests (unit and integration)
- `pytest tests/ -m "not gpu"` - excludes all MACE tests
- `pytest tests/ -m "not integration"` - runs all unit tests

## Citations
If you use StrainRelief or adapt the StrainRelief code for any purpose, please cite:

```bibtex
@misc{wallace2025strainrelief,
      title={Strain Problems got you in a Twist? Try StrainRelief: A Quantum-Accurate Tool for Ligand Strain Calculations},
      author={Ewan R. S. Wallace and Nathan C. Frey and Joshua A. Rackers},
      year={2025},
      eprint={2503.13352},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2503.13352},
}
```

```bibtex
@article{batatia2022mace,
  title={MACE: Higher order equivariant message passing neural networks for fast and accurate force fields},
  author={Batatia, Ilyes and Kovacs, David P and Simm, Gregor and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={11423--11436},
  year={2022}
}
```

## More information
For any questions, please reach out to [Ewan Wallace](https://www.linkedin.com/in/ewan-wallace-82297318a/): ewan.wallace@roche.com
