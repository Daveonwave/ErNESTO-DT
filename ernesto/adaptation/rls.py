import numpy as np


class RLS:
    """
    Recursive Least Squares (RLS).

    Stima ricorsivamente un vettore di parametri theta nel modello:

        y[k] = phi[k]^T theta

    ad ogni nuovo campione.

    Il metodo mantiene due quantità tra un campione e il successivo:

        theta -> stima corrente dei parametri
        P     -> matrice che rappresenta l'incertezza/informazione
                 associata alla stima.

    Per il nostro modello 2RC:
        theta = [phi1, phi2, b0, b1, b2, c]
    """

    def __init__(
        self,
        n_params,
        theta0=None,
        P0_scale=1e3,
        forgetting_factor=1.0
    ):
        """
        Inizializza l'algoritmo RLS.

        Parameters
        ----------
        n_params : int
            Numero di parametri da stimare.

        theta0 : array-like or None
            Stima iniziale dei parametri theta.
            Se None, viene usato un vettore di zeri.

        P0_scale : float
            Scala della matrice P iniziale:

                P[0] = P0_scale * I

            Un valore grande indica che all'inizio abbiamo
            poca informazione sui parametri e quindi siamo
            disposti a modificare molto la stima.

        forgetting_factor : float
            Forgetting factor lambda, con 0 < lambda <= 1.

                lambda = 1  -> nessun forgetting
                lambda < 1  -> maggiore peso ai dati recenti
        """

        # Numero di parametri da stimare.
        self.n = n_params


        if theta0 is None:
            # Se non viene fornita una stima iniziale,
            # partiamo da theta[0] = 0.
            self.theta0 = np.zeros(n_params)

        else:
            # Usiamo la stima iniziale fornita dall'utente.
            self.theta0 = np.asarray(theta0, dtype=float).copy()

        self.P0_scale = P0_scale

        # Forgetting factor lambda.
        self.lambda_ = forgetting_factor

        self.reset()  # Imposta theta e P alle condizioni iniziali

    def reset(self):
        """
        Riporta l'algoritmo RLS alle condizioni iniziali.

        Vengono azzerate tutte le informazioni accumulate
        durante l'identificazione:

            theta -> theta0
            P     -> P0_scale * I

        Il forgetting factor lambda e gli altri parametri
        di configurazione non vengono modificati.
        """

        # ---------------------------------------------------------
        # Inizializzazione della stima dei parametri
        # ---------------------------------------------------------
        self.theta = self.theta0.copy()

        # ---------------------------------------------------------
        # Inizializzazione della matrice P
        # ---------------------------------------------------------

        # P[0] = P0_scale * I
        #
        # I è la matrice identità.
        # Nel nostro caso, con 6 parametri:
        #
        # P[0] = P0_scale * I_6
        #
        # P0_scale grande -> maggiore "apertura" iniziale
        # della stima rispetto ai nuovi dati.

        self.P = self.P0_scale * np.eye(self.n)



    def update(self, phi, y):
        """
        Aggiorna la stima RLS usando un nuovo campione.

        Modello:

            y[k] = phi[k]^T theta

        Parameters
        ----------
        phi : array-like
            Vettore dei regressori phi[k].

            Nel nostro modello 2RC:

                phi[k] =
                [v[k-1], v[k-2],
                 i[k], i[k-1], i[k-2], 1]^T

        y : float
            Uscita misurata y[k].
            Nel nostro caso:

                y[k] = v[k]

        Returns
        -------
        theta : ndarray
            Nuova stima dei parametri theta[k].

        error : float
            Errore di predizione:

                e[k] = y[k] - phi[k]^T theta[k-1]

        K : ndarray
            Guadagno RLS K[k].
        """

        # Convertiamo phi in un array NumPy di tipo float.
        phi = np.asarray(phi, dtype=float)

        # =========================================================
        # 1. ERRORE DI PREDIZIONE
        # =========================================================
        #
        # Prima di usare il nuovo campione, abbiamo la stima:
        #
        #     theta[k-1]
        #
        # Con questa stima prevediamo:
        #
        #     y_hat[k] = phi[k]^T theta[k-1]
        #
        # L'errore tra misura e previsione è:
        #
        #     e[k] = y[k] - phi[k]^T theta[k-1]
        #
        error = y - phi @ self.theta

        # =========================================================
        # 2. DENOMINATORE DEL GUADAGNO RLS
        # =========================================================
        #
        # Formula teorica:
        #
        #     lambda + phi[k]^T P[k-1] phi[k]
        #
        # Questo termine è uno scalare e tiene conto di:
        #
        # - forgetting factor lambda
        # - informazione contenuta nel nuovo regressore phi[k]
        # - informazione già contenuta in P[k-1]
        #
        denominator = (
            self.lambda_
            + phi @ self.P @ phi
        )

        # =========================================================
        # 3. GUADAGNO RLS
        # =========================================================
        #
        # Formula:
        #
        #     K[k] =
        #         P[k-1] phi[k]
        #         -------------------------------
        #         lambda + phi[k]^T P[k-1] phi[k]
        #
        # K è un vettore con la stessa dimensione di theta.
        #
        # Indica in quali direzioni e di quanto correggere
        # la stima dei parametri dopo aver osservato l'errore.
        #
        K = (self.P @ phi) / denominator

        # =========================================================
        # 4. AGGIORNAMENTO DEI PARAMETRI
        # =========================================================
        #
        # Formula:
        #
        #     theta[k] =
        #         theta[k-1] + K[k] * e[k]
        #
        # Quindi:
        #
        #     nuova stima =
        #         vecchia stima + correzione
        #
        # La correzione dipende da:
        #
        # - quanto abbiamo sbagliato -> error
        # - quanto dobbiamo reagire -> K
        #
        self.theta = self.theta + K * error

        # =========================================================
        # 5. AGGIORNAMENTO DELLA MATRICE P
        # =========================================================
        #
        # Formula:
        #
        #     P[k] =
        #         1/lambda *
        #         (I - K[k] phi[k]^T) P[k-1]
        #
        # P viene aggiornata perché, dopo aver ricevuto
        # il nuovo campione, cambia anche la quantità di
        # informazione disponibile sulla stima dei parametri.
        #
        # np.outer(K, phi) costruisce:
        #
        #     K phi^T
        #
        # cioè una matrice n_params x n_params.
        #
        self.P = (
            (np.eye(self.n) - np.outer(K, phi))
            @ self.P
            / self.lambda_
        )

        # =========================================================
        # 6. RESTITUZIONE DEI RISULTATI
        # =========================================================
        #
        # Restituiamo:
        #
        #     theta -> nuova stima dei parametri
        #     error -> errore di predizione del campione
        #     K     -> guadagno RLS
        #
        # .copy() evita di restituire direttamente gli array
        # interni che verranno modificati alla prossima iterazione.
        #
        return self.theta.copy(), error, K.copy()