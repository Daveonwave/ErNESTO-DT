from src.online_learning.adaptation_utils.battery_adaptation import Battery_Adaptation



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

    #simulation = Simulation(alpha=0.01, batch_size=300, optimizer_method='Nelder-Mead',
    #                        training_window=12000, save_results= False, number_of_restarts = 1)
    #simulation.run_experiment()
    battery_adaptation = Battery_Adaptation(alpha=0.01, batch_size=300, optimizer_method='SLSQP',
                                            save_results= False, number_of_restarts = 3)
    battery_adaptation.run_experiment()





