ErneStoDT
==============================

Digital Twin of a Battery Energy Storage System

# Project Organization
------------
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is ...
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── digital_twin        <- Scripts containing the code of battery models
    |
    ├── tests               <- Testing code of this project.
       └── unit                 <- Tests on the single component of code (class, method or function)
       └── integration          <- Tests on multiple components of code interacting with each other
--------

<!--- 
# Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make preprocessing` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── preprocessing
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate preprocessing that has been transformed.
    │   ├── output      <- The final, canonical preprocessing sets for modeling.
    │   └── load            <- The original, immutable preprocessing dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── electrical             <- Trained and serialized electrical, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-preprocessing-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── preprocessing           <- Scripts to download or generate preprocessing
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn load preprocessing into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── electrical         <- Scripts to train electrical and then use trained electrical to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

---->
