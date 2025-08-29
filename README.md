# ErNESTO Digital Twin

This repository contains the implementation of the ENErgy STOrage (ErNESTO) Digital Twin. <br>
The framework is thought with a modular structure to run **_driven_** and **_scheduled_** experiments
of battery energy storage systems.

This project is a joint endeavor of [Politecnico di Milano](https://www.polimi.it) and [RSE](https://www.rse-web.it).

## :hammer_and_wrench: Installation

In order to use this codebase you need to work with a Python version >= 3.8.
To use ErNESTO, clone this repository and install the required libraries:

```bash
git clone https://github.com/Daveonwave/ErNESTO-DT.git && \
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
be retrieved by running `python ernesto.py --help` and `python ernesto.py [driven | scheduled] --help`
for the specific experiment.

Notice that `yaml` configuration files, contained in `data/config/` folder, have to adhere to a
standard formatting, validated within the script [schema.py](src/preprocessing/schema.py).
Follow the formatting of the already provided configuration file to generate new ones.

### Results visualization

To properly visualize results plots and metrics, to compare the simulated data to the provided ground truth, a friendly notebook has been provided within the `notebook` folder, stored at this [link](https://github.com/Daveonwave/ErNESTO-DT/blob/master/notebooks/evaluation_of_driven_exp.ipynb).

## :triangular_flag_on_post: Roadmap

The idea is to extend the DT to handle a more complex scenario, considering not only a single energy
storage system, but a broader Smart Grid. To reach this goal, further steps have to be taken. In particular:

1. **Parameter adaptation layer** (`online_learning` branch)
2. Integration within a **real-world energy storage system**, provided by the Italian research center [RSE](https://www.rse-web.it/en/).

Parallely, we are working to employ the DT also in other fields, such as:

1. **Reinforcement Learning**, by creating [ErNESTO-gym](https://github.com/Daveonwave/ErNESTO-gym), a [Gymnasium](https://gymnasium.farama.org) environment to test new RL algorithms
2. **Multi-Agent RL**, by improving [ErNESTO-gym](https://github.com/Daveonwave/ErNESTO-gym) in order to handle a distributed MiniGrid environment composed by multiple energy storage systems.

[comment]: <> (### Examples)

## :paperclip: Citing
```tex
@article{salaorni2025ernesto,
title = {A novel digital twin for battery energy storage systems in micro-grids},
journal = {Journal of Energy Storage},
volume = {132},
pages = {117745},
year = {2025},
issn = {2352-152X},
doi = {https://doi.org/10.1016/j.est.2025.117745},
url = {https://www.sciencedirect.com/science/article/pii/S2352152X25024582},
author = {Davide Salaorni and Federico Bianchi and Silvia Colnago and Andrea Barisione and Francesco Trovò and Marcello Restelli},
keywords = {Digital twin, Energy storage, Micro-grid}
}
```

