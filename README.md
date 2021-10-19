# Proper Value Equivalence

Code to reproduce the toy experiments illustrated in the Proper Value Equivalence [paper](https://arxiv.org/abs/2106.10316).

## Installation
Requires python >= 3.8
```bash
git clone git@github.com:chrisgrimm/proper_value_equivalence.git && \
cd proper_value_equivalence && \
python3.8 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt
```

## General
To generate the toy plots described in the paper one must first generate data using the ``main.py`` file and then visualize that data using the ``experiment_visualizations.ipynb`` notebook. 

The syntax for running the ``main.py`` file is as follows:
```
python main.py <experiment_type> <data_directory> <random_seed>
```
where ``experiment_type`` is either ``diameter`` or ``capacity``, ``data_directory`` is a folder where the data will be written to and ``random_seed`` is the seed for jax and numpy.


### Generating Data
To generate all data needed for the visualizations of toy experiments, run the following two commands (this may take a while):

```bash
python main.py diameter . 1234
```

```bash
python main.py capacity . 1234
```

When these commands are completed there will be two new directories ``ray_diameter`` and ``ray_capacity`` with the following structure
```bash
.
├── ray_diameter
│   └── run_experiment_XXXX          # Load and stress tests
└── ray_capacity
    └── run_experiment_YYYY
```
where ``XXXX`` and ``YYYY`` are replaced with date/time information corresponding to when you ran the experiment.

### Visualizing Data
To visualize the data, start up the Jupyter notebook by running
```bash
jupyter notebook
```
then open ```experiment_visualizations.ipynb``` in your browser. 

There is a cell labeled **Data loading configuration** which has three values to specify: ``RAY_DIR``, ``CAPACITY_EXPERIMENT_DIRS`` and ``DIAMETER_EXPERIMENT_DIRS``. Set ``RAY_DIR=.``, ``CAPACITY_EXPERIMENT_DIRS=YYYY`` and ``DIAMETER_EXPERIMENT_DIRS=XXXX`` where ``XXXX`` and ``YYYY`` are defined in the previous section.

This will save the plots in a folder ``./visuals/``





