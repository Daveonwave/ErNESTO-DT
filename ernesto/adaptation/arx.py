import numpy as np


# ============================================================
# 1. REGRESSORE ARX 1RC
# ============================================================

def build_regressor_1RC_at_k(v, i, k):
    """
    Costruisce il vettore dei regressori ARX 1RC al campione k.

    Modello 1RC:

        v[k] =
            phi1 * v[k-1]
            + b0 * i[k]
            + b1 * i[k-1]
            + c

    Quindi il regressore è:

        phi[k] =
        [v[k-1], i[k], i[k-1], 1]^T

    Parameters
    ----------
    v : array-like
        Serie temporale della tensione.

    i : array-like
        Serie temporale della corrente.

    k : int
        Indice del campione per cui costruire il regressore.
        Deve essere k >= 1.

    Returns
    -------
    phi : ndarray, shape (4,)
        Vettore dei regressori al campione k.
    """

    if k < 1:
        raise ValueError(
            "Per il modello ARX 1RC serve almeno un campione precedente."
        )

    return np.array([
        v[k - 1],      # v[k-1]
        i[k],          # i[k]
        i[k - 1],      # i[k-1]
        1.0            # termine costante c
    ], dtype=float)


# ============================================================
# 2. MATRICE DI REGRESSIONE ARX 1RC
# ============================================================

def build_regression_matrix_1RC(v, i):
    """
    Costruisce la matrice di regressione Phi e il vettore Y
    per il modello ARX 1RC.

    Per tutti i campioni k = 1, ..., N-1:

        phi[k] =
        [v[k-1], i[k], i[k-1], 1]^T

        Y[k] = v[k]

    La matrice risultante è:

        Phi =
        [phi[1]^T]
        [phi[2]^T]
        [   ...  ]
        [phi[N-1]^T]

    mentre:

        Y = [v[1], v[2], ..., v[N-1]]^T

    Questi oggetti possono essere usati direttamente per la
    soluzione Batch Least Squares:

        theta_hat = argmin ||Y - Phi theta||^2

    Parameters
    ----------
    v : array-like
        Serie temporale della tensione.

    i : array-like
        Serie temporale della corrente.

    Returns
    -------
    Phi : ndarray, shape (N-1, 4)
        Matrice dei regressori.

    Y : ndarray, shape (N-1,)
        Vettore delle uscite.
    """

    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)

    if len(v) != len(i):
        raise ValueError(
            "Tensione e corrente devono avere la stessa lunghezza."
        )

    if len(v) < 2:
        raise ValueError(
            "Servono almeno 2 campioni per costruire il regressore ARX 1RC."
        )

    N = len(v)

    # N-1 righe perché il primo regressore disponibile è k=1.
    #
    # Ogni riga contiene:
    #
    # [v[k-1], i[k], i[k-1], 1]
    Phi = np.zeros(
        (N - 1, 4),
        dtype=float
    )

    # Uscite corrispondenti:
    #
    # Y[k-1] = v[k]
    Y = np.zeros(
        N - 1,
        dtype=float
    )

    for k in range(1, N):

        Phi[k - 1] = build_regressor_1RC_at_k(
            v,
            i,
            k
        )

        Y[k - 1] = v[k]

    return Phi, Y


# ============================================================
# 3. REGRESSORE ARX 2RC
# ============================================================

def build_regressor_2RC_at_k(v, i, k):
    """
    Costruisce il vettore dei regressori ARX 2RC al campione k.

    Modello 2RC:

        v[k] =
            phi1 * v[k-1]
            + phi2 * v[k-2]
            + b0 * i[k]
            + b1 * i[k-1]
            + b2 * i[k-2]
            + c

    Quindi il regressore è:

        phi[k] =
        [v[k-1], v[k-2], i[k], i[k-1], i[k-2], 1]^T

    Parameters
    ----------
    v : array-like
        Serie temporale della tensione.

    i : array-like
        Serie temporale della corrente.

    k : int
        Indice del campione per cui costruire il regressore.
        Deve essere k >= 2.

    Returns
    -------
    phi : ndarray, shape (6,)
        Vettore dei regressori al campione k.
    """

    if k < 2:
        raise ValueError(
            "Per il modello ARX 2RC servono almeno due campioni precedenti."
        )

    return np.array([
        v[k - 1],      # v[k-1]
        v[k - 2],      # v[k-2]
        i[k],          # i[k]
        i[k - 1],      # i[k-1]
        i[k - 2],      # i[k-2]
        1.0            # termine costante c
    ], dtype=float)


# ============================================================
# 4. MATRICE DI REGRESSIONE ARX 2RC
# ============================================================

def build_regression_matrix_2RC(v, i):
    """
    Costruisce la matrice di regressione Phi e il vettore Y
    per il modello ARX 2RC.

    Per tutti i campioni k = 2, ..., N-1:

        phi[k] =
        [v[k-1], v[k-2], i[k], i[k-1], i[k-2], 1]^T

        Y[k] = v[k]

    La matrice risultante è:

        Phi =
        [phi[2]^T]
        [phi[3]^T]
        [   ...  ]
        [phi[N-1]^T]

    mentre:

        Y = [v[2], v[3], ..., v[N-1]]^T

    Questi oggetti possono essere usati direttamente per la
    soluzione Batch Least Squares:

        theta_hat = argmin ||Y - Phi theta||^2

    Parameters
    ----------
    v : array-like
        Serie temporale della tensione.

    i : array-like
        Serie temporale della corrente.

    Returns
    -------
    Phi : ndarray, shape (N-2, 6)
        Matrice dei regressori.

    Y : ndarray, shape (N-2,)
        Vettore delle uscite.
    """

    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)

    if len(v) != len(i):
        raise ValueError(
            "Tensione e corrente devono avere la stessa lunghezza."
        )

    if len(v) < 3:
        raise ValueError(
            "Servono almeno 3 campioni per costruire il regressore ARX 2RC."
        )

    N = len(v)

    # N-2 righe perché il primo regressore disponibile è k=2.
    #
    # Ogni riga contiene:
    #
    # [v[k-1], v[k-2], i[k], i[k-1], i[k-2], 1]
    Phi = np.zeros(
        (N - 2, 6),
        dtype=float
    )

    # Uscite corrispondenti:
    #
    # Y[k-2] = v[k]
    Y = np.zeros(
        N - 2,
        dtype=float
    )

    for k in range(2, N):

        Phi[k - 2] = build_regressor_2RC_at_k(
            v,
            i,
            k
        )

        Y[k - 2] = v[k]

    return Phi, Y

