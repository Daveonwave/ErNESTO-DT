from src.online_learning.adaptation.battery_adaptation import Battery_Adaptation

if __name__ == "__main__":

    battery_adaptation = Battery_Adaptation(alpha=0.01, batch_size=300, optimizer_method='L-BFGS-B',
                                            save_results= False, number_of_restarts = 3)
    battery_adaptation.run_experiment()





