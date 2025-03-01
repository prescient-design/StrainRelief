# StrainRelief
StrainRelief calculates the ligand strain of uncharged docked poses and has a suite of different force fields with which to do this. This includes a MACE neural network potential trained on SPICE2.

## Installation

From the root directory, run the following commands to install the package and its dependencies in editable mode:

```bash
mamba env create -f env.yml
mamba activate strain
pip install -e .

pre-commit install
```

## The Protocol

The protocol used in StrainRelief is designed to be simple, fast and model agnostic - all that is needed to apply a new force field is to write an ASE calculator wrapper. Additionally you can use any MACE model, such as these from the [MACE-OFF23](https://github.com/ACEsuit/mace-off/tree/main/mace_off23) repository.

The protocol consists of 5 steps:

1. Minimise the docked pose with a loose convergence criteria to give a local minimum.
2. Generate 20 conformers from the docked ligand pose.
3. Minimise the generated conformers (and the original docked pose) with a stricter convergence criteria.
4. Evaluate the energy of all conformers and choose the lowest energy as an approximation of the global minimum.
5. Calculate ligand strain, `E(local minimum) - E(global minimum)` and apply threshold.

**N.B.** energies returned are in kcal/mol.

## Usage
Choose a minimisation and energy evalation force field from `uff`, `mmff94`, `mmff94s`, `mace`.

The calculator works best when the same force field is used for both methods. If this is the case, `energy_eval` does not need to be specified.

```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s
```

```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@local_min=mmff94s \
    minimisation@global_min=mmff94s \
    energy_eval=mace \
    energy_eval.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model
```

```
strain-relief \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.output_file=data/example_ligboundconf_output.parquet \
    model=mace \
    model.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@local_min=mace \
    local_min.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    minimisation@global_min=mace \
    global_min.model_paths=s3://prescient-data-dev/strain_relief/models/MACE.model \
    local_min.fmax=0.50 \
    hydra.verbose=true
```

#### RDKit kwargs
The following dictionaries are passed directly to the function of that name.
- `conformers` (`EmbedMultipleConfs`)
- `minimisation.UFFOptimizeMoleculeConfs`
- `minimisation.MMFFGetMoleculeProperties`
- `minimisation.MMFFGetMoleculeForceField`
- `minimisation.Minimize`
- `energy_eval.MMFFGetMoleculeProperties`
- `energy_eval.MMFFGetMoleculeForceField`
- `energy_eval.UFFGetMoleculeForceField`

The hydra config is set up to allow additional kwargs to be passed to these functions e.g. `+minimisation.UFFOptimizeMoleculeConfs.vdwThresh=1.0`.

**Common kwargs**
- `threshold` (set by default to 16.1 kcal/mol - calibrated using LigBoundConf 2.0)
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

## More information
For any questions, please reach out to Ewan Wallace: ewan.wallace@roche.com
