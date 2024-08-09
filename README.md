# ErNESTO Digital Twin

This repository contains the implementation of the ENErgy STOrage (ErNESTO) Digital Twin. <br>
The framework is thought with a modular structure to run **_simulation_** and **_whatif_** experiments
of battery energy storage systems.

This project is a joint endeavor of [Politecnico di Milano](https://www.polimi.it) and [RSE](https://www.rse-web.it).

## :hammer_and_wrench: Installation

In order to use this codebase you need to work with a Python version >= 3.8.
To use ErNESTO, clone this repository and install the required libraries:

```bash
git clone https://github.com/Daveonwave/dt-rse.git && \
cd DT-rse/ && \
python -m pip install -r requirements.txt
```

## :battery: Usage

Before launching any script, add to the PYTHONPATH the root folder `DT-rse/`:

```bash
export PYTHONPATH=$(pwd)
```

### Reproducibility

To reproduce the experiments, a `bash` file must be run in the `scripts/` folder. For example:

```bash
./scripts/[run_check_up.sh | run_pv.sh | run_aging.sh]
```

Edit the `bash` files to choose different configuration files or models. The possible options can
be retrieved by running `python ernesto.py --help` and `python ernesto.py [simulation | whatif] --help`
for the specific experiment.

Notice that `yaml` configuration files, contained in `data/config/` folder, have to adhere to a
standard formatting, validated within the script [schema.py](src/preprocessing/schema.py).
Follow the formatting of the already provided configuration file to generate new ones.

### Results visualization

If experiments have been run with the `--plot` argument, the DT will automatically generate plots within
the output folder selected in the configuration file. Otherwise, it is possible to run experiments with
the `--save_results` flag to store the `csv` file within the output folder to get datasets of the
simulated data and _sanitized_ ground data.

## :triangular_flag_on_post: Roadmap

The idea is to extend the DT to handle a more complex scenario, considering not only a single energy
storage system, but a broader Smart Grid. To reach this goal, further steps have to be taken. In particular:

1. **Parameter adaptation layer** (`online_learning` branch)
2. **Wrapping with [Gymnasium](https://gymnasium.farama.org)** to apply Reinforcement Learning methods
3. **Build a multi-agent network**

[comment]: <> (### Examples)

## :paperclip: Citing

```

```
