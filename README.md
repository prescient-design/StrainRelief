# StrainRelief 💊
StrainRelief calculates the ligand strain of uncharged docked poses and has a suite of different force fields with which to do this. This includes our own MACE neural network potential trained on SPICE2 but also includes Meta's FairChem models such as e-SEN and UMA.

- 📄 The publication can be found [here](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00586).
- 📊 All relevant datasets [here](https://huggingface.co/datasets/erwallace/LigBoundConf2.0).
- 💬 RAG [chatbot](https://strain-relief.streamlit.app/) for questions about the paper and references.
- 💻 Chatbot source [code](https://github.com/erwallace/paper_query).
- 🐍 Published python [package](https://pypi.org/project/strain-relief/).

![Strain Relief Logo](assets/strain_relief_logo.png)

## Update: v0.5
1. Switched to uv for package management.
2. Introduced custom typing (`MolsDict`, `MolPropertiesDict`, `EnergiesDict` and `ConfEnergiesDict`) to make functions more readable.
3. Updated workflows and `MolsDict` to include charge and spin. This allows for charge aware NNPs such as eSEN and UMA. A boolean kwarg (`include_charged=True`) has been added to `load_parquets` to optionally filter these out.
4. Restructured calling of the main function to make it more intuitive with PyPi packaging.

## Installation
Pre-requisities: Python 3.11, PyTorch and PyTorch Geometric compatible with your envirnment

```bash
# For example
uv pip install torch==2.8.0 -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install torch-geometric==2.7.0 torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### Installation from PyPi

```
pip install strain-relief
```

### Installation from source (uv)
```bash
uv sync --extra dev --editable
```
or create a virtual environment and install the package and its dependencies in editable mode:
```bash
uv venv
source .venv/bin/activate

uv pip install -e ".[dev]"
uv pip install --force-reinstall e3nn==0.5 fairchem-core

uv run pre-commit install
```
**Note:** `mace-torch==0.3.x` requires `e3nn==0.4.4` (only for training, not inference). `fairchem-core` requires `e3nn>=0.5`. So until `mace-torch==0.4` is released we will have to do this finicky way of installing ([GitHub issue](https://github.com/ACEsuit/mace/issues/555)).

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

StrainRelief runs are configured using hydra configs.

### Python Package

```
from strain_relief import compute_strain

strains = compute_strain(poses: list[RDKit.Mol], config: DictConfig)

for i, r in computed.iterrows():
    print(f"Pose {r['id']} has a strain of {r['ligand_strain']:.2f} kcal/mol")
```
For a complete example see the tutorial [notebook](./examples/tutorial.ipynb).

### Command Line

```bash
strain-relief \
    experiment=mace \
    io.input.parquet_path=data/example_ligboundconf_input.parquet \
    io.output.parquet_path=data/example_ligboundconf_output.parquet \
    conformers.numConfs=1 \
```

More examples are given [here](./examples/examples.sh), including the command used for the calculations in the StrainRelief paper.

### Adding Your Own ASE Calculator

Add New ASE Calculator to `strain_relief/calculators/_nnp.py`:
```python
def another_calculator(model_paths: str, device: str, default_dtype: str, **kwargs: Any) -> Calculator:
    # set up your new ase calculator here
    return calculator
```
Add New Configs:
- `hydra_config/model/another_calculator.yaml`
- `hydra_config/energy_eval/another_calculator.yaml`
- `hydra_config/minimisation/another_calculator.yaml`
- `hydra_config/experiments` [Optional]

Update Method Dicts:
- `strain_relief.minimisation._minimisation.METHODS_DICT`
- `strain_relief.energy_eval._energy_eval.METHODS_DICT`
- `strain_relief.calculators.__init__.CALCULATORS_DICT`

### Configurations

**RDKit kwargs**

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
- `io.input.include_charged`
- `hydra.verbose`
- `seed`

### Logging

Logging is set to the `INFO` level by default which logs only aggregate information. `hydra.verbose=true` can be used to activate `DEBUG` level logging which includes information for every molecule and conformer.

## Unit Tests
- `uv run pytest tests/` - runs all tests (unit and integration)
- `uv run pytest tests/ -m "not gpu"` - excludes all MACE tests
- `uv run pytest tests/ -m "not integration"` - runs all unit tests

**NB** Tests requiring a FAIRChem model will be skipped if the OMol25 eSEN small conserving model is not located in `tests/models/eSEN.pt`. This model can be downloaded [here](https://huggingface.co/facebook/OMol25).

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

## More information
For any questions, please reach out to [Ewan Wallace](https://www.linkedin.com/in/ewan-wallace-82297318a/): ewan.wallace@roche.com
