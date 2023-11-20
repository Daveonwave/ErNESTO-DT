import numpy as np
from scipy.optimize import minimize

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
        self._actual_soc = 0.3 #I don't know how to start it!

        # Windows of current and voltage,
        self._i_batch = []
        self._v_batch = []

        self._v_hat = list()
        self._vocv = 0
        # TODO: Check if it has a meaning [ [lookup_table_vocv] vocv_t vocv_t+1 ... ], how should be _vocv_series
        self._vocv_series = list(lookup_table[:,0].flatten())
        #self._theta = np.zeros(5) #just for storing the output of the estimate_param
        self._params = {'r0': 0.01, 'r1': 0.01, 'c1': 0.01, 'soc_tau' : 0.00, 'q_max' : 0.00}

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
        print("------------actual_soc", self._actual_soc)
        self._vocv = np.interp(self._actual_soc, self._lookup_table[:,0], self._lookup_table[:,1])
        print("----------------self._vocv", self._vocv)
        self._vocv_series.append(self._vocv)

    def _get_soc(self):
        """

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
            print("-----------------vocv_series", len(self._vocv_series),self._vocv_series)

        if len(self._i_batch) < 2 or len(self._v_batch) < 2:
            pass
        else:
            self._vocv = (((1 / dt) + (1 / (c * rp))) * self._v_batch[-1] - (self._v_batch[-2] / dt) + (self._vocv / dt)
                           + (((rs / dt) + (1 / c) + (rs / (c * rp))) * self._i_batch[-1]) - (
                                       (rs / dt) * self._i_batch[-2])) / ((1 / dt) + (1 / (c * rp)))
            self._vocv_series.append(self._vocv)


    def _estimate_v(self, dt):
        """
        Args:
            dt ():
        """

        rs = self._params['r0'] * self._scale_factors['r0']
        rp = self._params['r1'] * self._scale_factors['r1']
        c = self._params['c1'] * self._scale_factors['c1']

        if self._first_estimation_v:
            self._v_hat.append(self._v_batch[-1])
            self._first_estimation_v = False

        #TODO: MUST BE FIXED !!!
        tmp = ((1 / dt) * self._v_hat[-1] + ((1 / dt) + 1 / (c * rp)) * self._vocv_series[-1] - (
                            1 / dt) * self._vocv_series[-2] - ((rs / dt) + (1 / c) * (rs / (rp * c))) * self._i_batch[-1] + (rs / dt) * self._i_batch[-2]) / ((1 / dt) + 1 / (c * rp))
        self._v_hat.append(tmp)






    def _loss_function(self,theta , dt):
        """
        theta is the vector of parameter that must be optimized
        """
        # TODO: Check how v_batch increases and if it matches with v_est
        #self._actual_soc = self._actual_soc - np.sum(np.array(self._i_batch)) * (dt / self._capacity)
        # TODO: HERE THERE IS THE PROBLEM
        #self._get_vocv()

        self._params['r0'] = theta[0]
        self._params['r1'] = theta[1]
        self._params['c1'] = theta[2]
        self._params['soc_tau'] = theta[3]
        self._params['q_max'] = theta[4]


        #self._estimate_v(dt)

        print("len of v_hat", len(self._v_hat))
        v_est = np.array(self._v_hat)
        print("Len of v_batch", len(self._v_batch))
        #print("Len of v_est", len(v_est))
        v = np.array(self._v_batch)
        loss = 0.

        # TODO: Understand why they are not of the same lenght -> answer: v it's not incremented, RMK: THE FIRST ITERATION WORKS!!!
        #print( "   v_est",v_est)
        #print("   v", v)

        if len(v_est) == len(v) and len(v_est) >= 1 and len(v) >= 1:
            loss = np.sum(np.abs(v_est - v))
            print("this is the loss ye ye: ",loss)
            print("\n")

        return loss


    def _estimate_params(self, dt):
        """
        Estimation of parameters through a nonlinear optimization approach.
        Returns: r0, r1, c, soc_tau, q_max
        """
        loss = lambda x: self._loss_function(x, dt)

        # TODO: Implement the multiple restarts
        theta0 = np.random.rand(len(self._params))

        #TODO: HERE THE IDEA IS TO CHECK IF THE VOCV_SERIES AND THE V_HAT DOESN'T GROW
        self._actual_soc = self._actual_soc - np.sum(np.array(self._i_batch)) * (dt / self._capacity)
        self._get_vocv()
        self._estimate_v(dt)

        res = minimize(loss, theta0, method='BFGS', options={'disp': True, 'maxiter':1})

        self._params['ro'] = res.x[0]
        self._params['r1'] = res.x[1]
        self._params['c1'] = res.x[2]
        self._params['soc_tau'] = res.x[3]
        self._params['q_max'] = res.x[4]




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
            self._actual_soc = self._actual_soc - (1 / self._capacity) * i * dt
            # TODO: CORRECT THIS DOUBLE EXTEND!!!
            self._extend_window(i=i, v=v)
            if len(self._v_batch) == 1 and len(self._i_batch):
                self._extend_window(i,v)

            print("\n the value for the if: _max_soc - _min_soc Soc_time_window",self._max_soc - self._min_soc, self._soc_time_window )
            if (self._max_soc - self._min_soc) < self._soc_time_window:

                self._estimate_params(dt=dt)
                print(" the param are:", self._params)
                self._estimate_vocv(dt)
                self._get_soc()

                #self._max_soc = 0.
                #self._min_soc = 0.
                #self._i_batch = list(self._i_batch[-1])
                #self._v_batch = list(self._v_batch[-1])
            else:
                self._first_window = False
                print("the flag is false!")
        else:
            #interpolazione
            self._extend_window(i=i, v=v)
            #self._estimate_vocv(dt)
            #self._get_soc()

            # flag only first iteration
            if (self._max_soc - self._min_soc) < self._moving_step:
                print("**********************************************************************************************************************Im here!!")
                self._estimate_params(dt=dt)

                self._estimate_vocv(dt)
                self._get_soc()

                #self._max_soc = 0.
                #self._min_soc = 0.
                #self._i_batch = list(self._i_batch[-1])
                #self._v_batch = list(self._v_batch[-1])

            # TODO: continuare e capire quando sovrascrivere i_batch e v_batch e reinizializzarli per nuova window

        return self._actual_soc








