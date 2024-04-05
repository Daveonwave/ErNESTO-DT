import numpy as np
from scipy.optimize import minimize
import nevergrad as ng

from scipy.interpolate import interp1d

class VDBSE:
    """

    """
    def __init__(self,
                 capacity: float,
                 soc_time_window: float,
                 moving_step: float,
                 restarts: int,
                 lookup_table,
                 scale_factor: dict
                 ):
        """
        Args:
            capacity ():
            soc_time_window ():
            moving_step ():
            restarts ():
            lookup_table ():
            scale_factor (dict):
        """
        self._capacity = capacity
        self._soc_time_window = soc_time_window
        self._moving_step = moving_step
        self._restarts = restarts
        # TODO: Since lookup_table is read from a csv in external piece of code it is assumed to be already a numpy array
        self._lookup_table = lookup_table
        self._scale_factors = scale_factor
        self._verbose = False

        self._max_soc = 0.
        self._min_soc = 0.
        self._actual_soc = 0.4266 #I don't know how to start it!

        # Windows of current and voltage,
        self._i_batch = []
        self._v_batch = []

        self._v_hat_batch = []
        self._vocv = 0
        # TODO: Check if it has a meaning [ [lookup_table_vocv] vocv_t vocv_t+1 ... ], how should be _vocv_series
        self._vocv_batch = list(lookup_table[:, 0].flatten())
        theta = np.random.rand(5) #just for storing the output of the estimate_param
        self._params = {'r0': theta[0], 'r1': theta[1], 'c1': theta[2], 'soc_tau' : theta[3], 'q_max' : theta[4]}

        self._first_window = True
        self._first_estimation_v = True

    def _extend_window(self, i, v):
        """
        We are not using the window indices, but we directly store i and v in batches

        Args:
            i (float):
            v (float):
        """

        self._max_soc = max(self._max_soc, self._actual_soc)
        self._min_soc = min(self._min_soc, self._actual_soc)
        self._i_batch.append(i)
        self._v_batch.append(v)

    def _get_vocv(self):
        """
        Take the lookup table and the soc estimated (soC_tau ), then yields the vocv value by interpolation.
        It is needed for the first estimation of the vocv, because of the presence of vocv(t-1)
        self._lookup_table[:,0] : Vocv and self._lookup_table[:,1] : SoC
        """
        self._vocv = np.interp(self._actual_soc, self._lookup_table[:,0], self._lookup_table[:,1], left=np.nan, right=np.nan)
        self._vocv_batch.append(self._vocv)

    def _interpolate_soc(self):
        """
        spline_interpolation
        """
        self._actual_soc = np.interp(self._vocv, self._lookup_table[:, 1], self._lookup_table[:, 0])



    def _estimate_vocv(self, dt):
        """

        Args:
            dt (): sampling time passed by the estimator class

        Returns:

        """
        c = self._params['r0'] * self._scale_factors['r0']
        rp = self._params['r1'] * self._scale_factors['r1']
        rs = self._params['c1'] * self._scale_factors['c1']

        # TODO: check: when is performed the first estimation which value of Vocv must be exploited??0 ---- get_Vocv(SoC_tau_est,lookup)
        if self._first_window:
            self._get_vocv()
            print("-----------------vocv_series", len(self._vocv_batch), self._vocv_batch)

        if len(self._i_batch) < 2 or len(self._v_batch) < 2:
            pass
        else:
            self._vocv = (((1 / dt) + (1 / (c * rp))) * self._v_batch[-1] - (self._v_batch[-2] / dt) + (self._vocv / dt)
                           + (((rs / dt) + (1 / c) + (rs / (c * rp))) * self._i_batch[-1]) - (
                                       (rs / dt) * self._i_batch[-2])) / ((1 / dt) + (1 / (c * rp)))
            self._vocv_batch.append(self._vocv)


    def _estimate_v(self, dt):
        """
        Args:
            dt ():
        """

        rs = self._params['r0'] * self._scale_factors['r0']
        rp = self._params['r1'] * self._scale_factors['r1']
        c = self._params['c1'] * self._scale_factors['c1']

        if self._first_estimation_v:
            self._v_hat_batch.append(self._v_batch[0])
            self._first_estimation_v = False

        for k in range(len(self._v_batch)):
            v_hat = ((1 / dt) * self._v_hat_batch[k] + ((1 / dt) + 1 / (c * rp)) * self._vocv_batch[k] - (
                            1 / dt) * self._vocv_batch[k - 1] - ((rs / dt) + (1 / c) * (rs / (rp * c))) * self._i_batch[k - 1] + (rs / dt) * self._i_batch[k - 1]) / ((1 / dt) + 1 / (c * rp))
            self._v_hat_batch.append(v_hat)






    def _loss_function(self,theta , dt):
        """
        theta is the vector of parameter that must be optimized
        """

        self._params['r0'] = theta[0]
        self._params['r1'] = theta[1]
        self._params['c1'] = theta[2]
        self._params['soc_tau'] = theta[3]
        self._params['q_max'] = theta[4]


        #self._estimate_v(dt)

        #print("len of v_hat", len(self._v_hat_batch))
        v_est = np.array(self._v_hat_batch)
        #print("Len of v_batch", len(self._v_batch))
        #print("Len of v_est", len(v_est))
        v = np.array(self._v_batch)
        #loss = 0.


        if len(v_est) == len(v) and len(v_est) >= 1 and len(v) >= 1:
            loss = np.sum(np.abs(v_est - v))
            print("this is the loss ye ye: ",loss)
            print("\n")
        else:
            v_est = v_est[-len(v):]
            #print("len of v_est after cutting", len(v_est))
            loss = np.sum(np.abs(v_est - v))
            print("this is the loss ye ye: ", loss)
            print("\n")

        return loss


    def _estimate_params(self, dt):
        """
                Estimation of parameters through a nonlinear optimization approach.
                Returns: r0, r1, c, soc_tau, q_max
                """

        loss = lambda x: self._loss_function(x, dt)
        self._actual_soc = abs(self._actual_soc - np.sum(np.array(self._i_batch)) * (dt / self._capacity))
        self._get_vocv()
        self._estimate_v(dt)

        optimizer = ng.optimizers.NGOpt(parametrization=5, budget=100)
        recommendation = optimizer.minimize(loss)
        print("reccomendation is:")
        print(recommendation.value)

        self._params['ro'] = abs(recommendation.value[0]) * self._scale_factors['r0']
        self._params['r1'] = abs(recommendation.value[1]) * self._scale_factors['r1']
        self._params['c1'] = abs(recommendation.value[2]) * self._scale_factors['c1']
        self._params['soc_tau'] = abs(recommendation.value[3])
        self._params['q_max'] = abs(recommendation.value[4])

    def estimate_soc(self, i, v, dt):
        """
        Core function of the algorithm.

        Args:
            i (float): sample of I at time t
            v (float): sample of V at time t
            dt (float): time delta from previous sample

        Returns: soc estimation

        """
        if self._first_window:
            #print("\n the value for the if: _max_soc - _min_soc Soc_time_window",self._max_soc - self._min_soc, self._soc_time_window )
            if (self._max_soc - self._min_soc) < self._soc_time_window:
                self._actual_soc = abs(self._actual_soc - (1 / self._capacity) * i * dt)
                self._extend_window(i=i, v=v)
                #if len(self._v_batch) == 1 and len(self._i_batch) == 1:
                #    self._extend_window(i, v)

            else:
                self._first_window = False
                print("the flag is false!")

                self._estimate_params(dt=dt)
                print(" the param are:", self._params)
                self._estimate_vocv(dt)
                # TODO: JUST A TRIAL
                self._interpolate_soc()

                self._max_soc = 0.
                self._min_soc = 0.
                self._i_batch = self._i_batch[-2:]
                self._v_batch = self._v_batch[-2:]
                self._v_hat_batch = self._v_hat_batch[-2:]

        else:
            if (self._max_soc - self._min_soc) < self._moving_step:
                print("moving step passed!")
                #self._actual_soc = abs(self._actual_soc - (1 / self._capacity) * i * dt)
                self._interpolate_soc()
                self._extend_window(i=i, v=v)

            else:
                self._estimate_vocv(dt)
                self._estimate_params(dt=dt)

                self._max_soc = 0.
                self._min_soc = 0.
                self._i_batch = self._i_batch[-2:]
                self._v_batch = self._v_batch[-2:]
                self._v_hat_batch = self._v_hat_batch[-2:]

            self._interpolate_soc()

        return self._actual_soc








