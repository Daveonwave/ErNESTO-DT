import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ernesto.adaptation.arx import build_regressor_1RC_at_k
from ernesto.adaptation.rls import RLS


# ============================================================
# CONFIGURAZIONE
# ============================================================

Ts = 1.0


# ============================================================
# 1. ARX-RLS 1RC
# ============================================================

def run_rls_1RC(
    voltage,
    current,
    forgetting_factor,
    P0_SCALE=1e9
):
    """
    Esegue RLS per il modello ARX di un ECM 1RC.

    Modello:

        v[k] =
            phi1 * v[k-1]
            + b0 * i[k]
            + b1 * i[k-1]
            + c

    Vettore dei parametri:

        theta =
        [phi1, b0, b1, c]

    Regressore:

        phi[k] =
        [v[k-1], i[k], i[k-1], 1]^T

    Restituisce:

        theta_final
        theta_history
        error_history
    """

    # --------------------------------------------------------
    # Inizializzazione RLS
    # --------------------------------------------------------

    rls = RLS(
        n_params=4,
        theta0=np.zeros(4),
        P0_scale=P0_SCALE,
        forgetting_factor=forgetting_factor
    )

    # --------------------------------------------------------
    # Allocazione memoria
    # --------------------------------------------------------

    n = len(voltage) - 1

    theta_history = np.empty(
        (n, 4),
        dtype=float
    )

    error_history = np.empty(
        n,
        dtype=float
    )

    # --------------------------------------------------------
    # RLS ricorsivo
    # --------------------------------------------------------

    for k in range(
        1,
        len(voltage)
    ):

        # ----------------------------------------------------
        # Regressore 1RC
        #
        # phi[k] =
        # [v[k-1], i[k], i[k-1], 1]^T
        # ----------------------------------------------------

        phi = np.array([
            voltage[k - 1],
            current[k],
            current[k - 1],
            1.0
        ])

        # ----------------------------------------------------
        # Aggiornamento RLS
        # ----------------------------------------------------

        theta, error, _ = rls.update(
            phi,
            voltage[k]
        )

        idx = k - 1

        theta_history[idx] = theta
        error_history[idx] = error

    return {
        "theta_final": rls.theta.copy(),
        "theta_history": theta_history,
        "error_history": error_history
    }


# ============================================================
# 2. ARX -> PARAMETRI FISICI 1RC
# ============================================================

def theta_to_physical_1RC(
    theta,
    Ts
):
    """
    Converte i coefficienti ARX 1RC:

        theta =
        [phi1, b0, b1, c]

    nei parametri fisici:

        R0
        R1
        C1
        OCV
        tau1
        alpha

    Relazioni:

        alpha = phi1

        R0 = -b0

        R1 =
            (R0*alpha - b1)
            / (1-alpha)

        tau1 =
            -Ts / log(alpha)

        C1 =
            tau1 / R1

        OCV =
            c / (1-alpha)

    Restituisce valid=False se i parametri
    non sono fisicamente ammissibili.
    """

    # --------------------------------------------------------
    # Estrazione coefficienti ARX
    # --------------------------------------------------------

    phi1, b0, b1, c = theta

    # --------------------------------------------------------
    # Polo discreto
    # --------------------------------------------------------

    alpha = phi1

    # Per un ramo RC fisico:

    #     0 < alpha < 1

    if not (
        0.0 < alpha < 1.0
    ):
        return {
            "valid": False,
            "reason": (
                f"Invalid alpha = {alpha}"
            )
        }

    # --------------------------------------------------------
    # R0
    # --------------------------------------------------------

    R0 = -b0

    if not np.isfinite(R0) or R0 <= 0:
        return {
            "valid": False,
            "reason": (
                f"Invalid R0 = {R0}"
            )
        }

    # --------------------------------------------------------
    # R1
    # --------------------------------------------------------

    denominator_R1 = (
        1.0 - alpha
    )

    if abs(denominator_R1) < 1e-12:
        return {
            "valid": False,
            "reason": (
                "alpha too close to 1."
            )
        }

    R1 = (
        R0 * alpha
        - b1
    ) / denominator_R1

    if not np.isfinite(R1) or R1 <= 0:
        return {
            "valid": False,
            "reason": (
                f"Invalid R1 = {R1}"
            )
        }

    # --------------------------------------------------------
    # Costante di tempo
    # --------------------------------------------------------

    tau1 = (
        -Ts
        / np.log(alpha)
    )

    if not np.isfinite(tau1) or tau1 <= 0:
        return {
            "valid": False,
            "reason": (
                f"Invalid tau1 = {tau1}"
            )
        }

    # --------------------------------------------------------
    # C1
    # --------------------------------------------------------

    C1 = tau1 / R1

    if not np.isfinite(C1) or C1 <= 0:
        return {
            "valid": False,
            "reason": (
                f"Invalid C1 = {C1}"
            )
        }

    # --------------------------------------------------------
    # OCV
    # --------------------------------------------------------

    OCV = (
        c
        / denominator_R1
    )

    if not np.isfinite(OCV):
        return {
            "valid": False,
            "reason": (
                f"Invalid OCV = {OCV}"
            )
        }

    # --------------------------------------------------------
    # Risultato
    # --------------------------------------------------------

    return {
        "valid": True,

        "R0": R0,
        "R1": R1,
        "C1": C1,
        "OCV": OCV,

        "tau1": tau1,
        "alpha": alpha
    }


# ============================================================
# 3. CONVERSIONE DELL'INTERA STORIA
#    ARX -> PARAMETRI FISICI
# ============================================================

def convert_history_to_physical_1RC(
    theta_history,
    time,
    Ts,
    decimation=1
):
    """
    Converte la storia temporale ARX 1RC nei parametri fisici.

    decimation:
        numero di campioni saltati tra due conversioni.

    Per esempio:

        decimation=100

    converte un punto ogni 100 campioni.
    """

    indices = np.arange(
        0,
        len(theta_history),
        decimation
    )

    # La storia theta[0] corrisponde al campione k=1.
    physical_time = time[
        indices + 1
    ]

    n = len(indices)

    # --------------------------------------------------------
    # Array di output
    # --------------------------------------------------------

    R0 = np.full(
        n,
        np.nan
    )

    R1 = np.full(
        n,
        np.nan
    )

    C1 = np.full(
        n,
        np.nan
    )

    OCV = np.full(
        n,
        np.nan
    )

    tau1 = np.full(
        n,
        np.nan
    )

    alpha = np.full(
        n,
        np.nan
    )

    valid = np.zeros(
        n,
        dtype=bool
    )

    # --------------------------------------------------------
    # Conversione
    # --------------------------------------------------------

    for j, idx in enumerate(indices):

        physical = theta_to_physical_1RC(
            theta_history[idx],
            Ts
        )

        if not physical["valid"]:
            continue

        valid[j] = True

        R0[j] = physical["R0"]
        R1[j] = physical["R1"]
        C1[j] = physical["C1"]
        OCV[j] = physical["OCV"]
        tau1[j] = physical["tau1"]
        alpha[j] = physical["alpha"]

    return {
        "time": physical_time,

        "valid": valid,

        "R0": R0,
        "R1": R1,
        "C1": C1,
        "OCV": OCV,

        "tau1": tau1,
        "alpha": alpha
    }

