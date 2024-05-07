from src.digital_twin.bess import BatteryEnergyStorageSystem
from scipy.optimize import minimize
from notebooks.online_learning.grid import Grid
from notebooks.online_learning.optimizer import Optimizer
from notebooks.online_learning.simulation_loop import Simulation
import pandas as pd
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #cli -> check why doesn't work
    """
    parser = argparse.ArgumentParser(description="Run simulation experiment")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha value")
    parser.add_argument("--batch-size", type=int, default=400, help="Batch size")
    parser.add_argument("--training-window", type=int, default=10000, help="Training window size")
    args = parser.parse_args()
    simulation = Simulation( alpha=args.alpha, batch_size=args.batch_size, optimizer = None, training_window=args.training_window)
    """

    simulation = Simulation(alpha=0.3, batch_size=100, optimizer=None, training_window=20000)

    simulation.run_experiment()




