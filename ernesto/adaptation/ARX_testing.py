import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arx import build_regressor_at_k
from rls import RLS


# ============================================================
# CONFIGURAZIONE
# ============================================================

# CSV_FILE = "Nuova_prova_OL.csv"
CSV_FILE = "C1_5_Cluster_OL.csv"



# Nel file corrente il segmento selezionato parte da qui.
# START_ROW = 69716
START_ROW = 0



# Campionamento nominale ricavato da DWell Time
# Ts = 0.1  # [s]
Ts = 1  # [s]


# Forgetting factors da confrontare
LAMBDA_VALUES = [
    0.999,
]

# Inizializzazione RLS
P0_SCALE_VALUES = [
    1e9,
]

# Conversione ARX -> fisico:
# non serve farla su tutti i campioni.
# 100 campioni = 10 s
PHYSICAL_DECIMATION = 1

# Finestra di smoothing per i plot.
# Deve essere più lunga del periodo delle oscillazioni visibili.
SMOOTH_WINDOW_SECONDS = 1800

# Mostra anche il segnale grezzo sotto quello filtrato.
SHOW_RAW_IN_FILTERED_PLOTS = True

# Ampiezza minima del cambio corrente per identificare un nuovo tratto/ciclo.
CURRENT_STEP_THRESHOLD = 0.05  # [A]

# Numero minimo di campioni per accettare un segmento di corrente.
MIN_SEGMENT_SAMPLES = 20


# Validazione Digital Twin
VALIDATION_WARMUP_SECONDS = 200000
VALIDATION_END_TRIM_SECONDS = 0

# Soglia per separare carica/scarica nelle metriche.
CURRENT_SIGN_THRESHOLD = 1.0  # [A]

# Soglie temperatura per metriche separate.
LOW_TEMPERATURE_THRESHOLD = 20.0   # [°C]
HIGH_TEMPERATURE_THRESHOLD = 35.0  # [°C]

# Finestra rolling RMS errore tensione.
VALIDATION_RMS_WINDOW_SECONDS = 1800



# ============================================================
# 1. LETTURA E PREPROCESSING
# ============================================================

def load_experiment_csv(filename):
    """
    Carica il CSV e restituisce:

        time     [s]
        voltage  [V]
        current  [A]

    La corrente viene cambiata di segno per adottare
    la convenzione del modello:

        i > 0  -> scarica
        i < 0  -> carica

    Nel CSV originale:
        i > 0  -> carica
        i < 0  -> scarica
    """

    df = pd.read_csv(
        filename,
        skiprows=0
    )

    # Manteniamo il segmento scelto.
    # df = df.iloc[START_ROW:START_ROW+1500].reset_index(drop=True)
    df = df.iloc[START_ROW:].reset_index(drop=True)

    # Pulizia nomi colonne
    df.columns = df.columns.str.strip()

    # required_columns = [
    #     "DWell Time(ms)",
    #     "Voltage(V)",
    #     "Current(A)"
    # ]

    # for column in required_columns:
    #     if column not in df.columns:
    #         raise ValueError(
    #             f"Colonna '{column}' non trovata.\n"
    #             f"Colonne disponibili:\n{df.columns.tolist()}"
    #         )

    # --------------------------------------------------------
    # Tempo
    # --------------------------------------------------------

    # dwell_ms = pd.to_numeric(
    #     df["DWell Time(ms)"],
    #     errors="coerce"
    # ).to_numpy(dtype=float)

    # dwell_ms -= dwell_ms[0]

    # time = dwell_ms / 1000.0

    # time = pd.to_numeric(
    #     df["Time [s]"],
    #     errors="coerce"
    # ).to_numpy(dtype=float)

    time = np.arange(len(df.index), dtype=float)

    # --------------------------------------------------------
    # Tensione
    # --------------------------------------------------------

    # voltage = pd.to_numeric(
    #     df["Voltage(V)"],
    #     errors="coerce"
    # ).to_numpy(dtype=float)

    voltage = pd.to_numeric(
        df["Voltage [V]"],
        errors="coerce"
    ).to_numpy(dtype=float)

    # --------------------------------------------------------
    # Corrente
    # --------------------------------------------------------

    # current_csv = pd.to_numeric(
    #     df["Current(A)"],
    #     errors="coerce"
    # ).to_numpy(dtype=float)

    current_csv = pd.to_numeric(
        df["Current [A]"],
        errors="coerce"
    ).to_numpy(dtype=float)

    # Convenzione del modello:
    #
    # CSV:
    #   + -> carica
    #   - -> scarica
    #
    # Modello:
    #   + -> scarica
    #   - -> carica
    # current = -current_csv

    temperature = pd.to_numeric(
        df["Cell Temperature [degC]"],
        errors="coerce"
    ).to_numpy(dtype=float)

    # --------------------------------------------------------
    # Rimozione eventuali campioni non validi
    # --------------------------------------------------------

    valid = (
        np.isfinite(time)
        & np.isfinite(voltage)
        & np.isfinite(current)
        & np.isfinite(temperature)
    )

    time = time[valid]
    voltage = voltage[valid]
    current = current[valid]
    temperature = temperature[valid]


    if len(time) < 3:
        raise ValueError("Troppi pochi campioni validi.")

    return time, voltage, current, temperature


# ============================================================
# 2. CONTROLLO CAMPIONAMENTO
# ============================================================

def check_sampling(time):
    """
    Verifica il campionamento effettivo.
    """

    dt = np.diff(time)

    print("\nSampling:")
    print(f"  mean   dt = {np.mean(dt):.9f} s")
    print(f"  median dt = {np.median(dt):.9f} s")
    print(f"  min    dt = {np.min(dt):.9f} s")
    print(f"  max    dt = {np.max(dt):.9f} s")

    if not np.allclose(
        dt,
        Ts,
        rtol=1e-6,
        atol=1e-9
    ):
        print(
            "\nATTENZIONE: alcuni intervalli differiscono da Ts."
        )

    return dt


# ============================================================
# 3. RLS
# ============================================================

def run_rls(
    voltage,
    current,
    forgetting_factor,
    P0_SCALE=1e9
):
    """
    Esegue ARX-RLS su tutti i campioni.

    Restituisce:

        theta_history
        error_history
    """

    rls = RLS(
        n_params=6,
        theta0=np.zeros(6),
        P0_scale=P0_SCALE,
        forgetting_factor=forgetting_factor
    )

    theta_history = np.empty(
        (len(voltage) - 2, 6)
    )

    error_history = np.empty(
        len(voltage) - 2
    )

    for k in range(2, len(voltage)):

        phi = build_regressor_at_k(
            voltage,
            current,
            k
        )

        theta, error, _ = rls.update(
            phi,
            voltage[k]
        )

        theta_history[k - 2] = theta
        error_history[k - 2] = error

    return {
        "theta_final": rls.theta.copy(),
        "theta_history": theta_history,
        "error_history": error_history
    }


# ============================================================
# 4. ARX -> PARAMETRI FISICI
# ============================================================

def theta_to_physical(theta, Ts):
    """
    Converte:

        theta =
        [phi1, phi2, b0, b1, b2, c]

    nei parametri fisici 2RC.

    Restituisce valid=False se il modello ARX non
    corrisponde a un 2RC fisicamente ammissibile.
    """

    phi1, phi2, b0, b1, b2, c = theta

    # --------------------------------------------------------
    # R0
    # --------------------------------------------------------

    R0 = -b0

    # --------------------------------------------------------
    # Poli discreti
    # --------------------------------------------------------

    roots = np.roots([
        1.0,
        -phi1,
        -phi2
    ])

    # Devono essere reali
    if np.any(np.abs(roots.imag) > 1e-8):
        return {
            "valid": False
        }

    roots = roots.real

    # Convenzione:
    # alpha1 > alpha2
    alpha1, alpha2 = np.sort(roots)[::-1]

    # Per un ramo RC:
    #
    #     0 < alpha < 1
    #
    if not (
        0.0 < alpha2 < alpha1 < 1.0
    ):
        return {
            "valid": False
        }

    # --------------------------------------------------------
    # Costanti di tempo
    # --------------------------------------------------------

    tau1 = -Ts / np.log(alpha1)
    tau2 = -Ts / np.log(alpha2)

    # --------------------------------------------------------
    # Recupero R1, R2
    # --------------------------------------------------------

    K1 = (
        R0 * (alpha1 + alpha2)
        - b1
    )

    K2 = (
        b2
        + R0 * alpha1 * alpha2
    )

    denominator_R1 = (
        (1.0 - alpha1)
        * (alpha2 - alpha1)
    )

    denominator_R2 = (
        (1.0 - alpha2)
        * (alpha1 - alpha2)
    )

    # Poli quasi coincidenti
    if (
        abs(denominator_R1) < 1e-12
        or abs(denominator_R2) < 1e-12
    ):
        return {
            "valid": False
        }

    R1 = (
        K2 - alpha1 * K1
    ) / denominator_R1

    R2 = (
        K2 - alpha2 * K1
    ) / denominator_R2

    # --------------------------------------------------------
    # Capacità
    # --------------------------------------------------------

    C1 = tau1 / R1
    C2 = tau2 / R2

    # --------------------------------------------------------
    # OCV
    # --------------------------------------------------------

    denominator_ocv = (
        (1.0 - alpha1)
        * (1.0 - alpha2)
    )

    OCV = c / denominator_ocv

    # --------------------------------------------------------
    # Controllo fisico
    # --------------------------------------------------------

    if (
        R0 <= 0
        or R1 <= 0
        or R2 <= 0
        or C1 <= 0
        or C2 <= 0
    ):
        return {
            "valid": False
        }

    return {
        "valid": True,
        "R0": R0,
        "R1": R1,
        "C1": C1,
        "R2": R2,
        "C2": C2,
        "OCV": OCV,
        "tau1": tau1,
        "tau2": tau2,
        "alpha1": alpha1,
        "alpha2": alpha2
    }


# ============================================================
# 5. CONVERSIONE DELL'INTERA STORIA IN PARAMETRI FISICI
# ============================================================

def convert_history_to_physical(
    theta_history,
    time,
    Ts,
    decimation
):
    """
    Converte la storia ARX in parametri fisici su una
    griglia temporale diradata.

    Questo evita di eseguire np.roots() su ogni singolo
    campione di un dataset molto grande.
    """

    indices = np.arange(
        0,
        len(theta_history),
        decimation
    )

    physical_time = time[
        indices + 2
    ]

    n = len(indices)

    R0 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    C1 = np.full(n, np.nan)
    R2 = np.full(n, np.nan)
    C2 = np.full(n, np.nan)
    OCV = np.full(n, np.nan)
    tau1 = np.full(n, np.nan)
    tau2 = np.full(n, np.nan)

    valid = np.zeros(
        n,
        dtype=bool
    )

    for j, idx in enumerate(indices):

        physical = theta_to_physical(
            theta_history[idx],
            Ts
        )

        if not physical["valid"]:
            continue

        valid[j] = True

        R0[j] = physical["R0"]
        R1[j] = physical["R1"]
        C1[j] = physical["C1"]
        R2[j] = physical["R2"]
        C2[j] = physical["C2"]
        OCV[j] = physical["OCV"]
        tau1[j] = physical["tau1"]
        tau2[j] = physical["tau2"]

    return {
        "time": physical_time,
        "valid": valid,
        "R0": R0,
        "R1": R1,
        "C1": C1,
        "R2": R2,
        "C2": C2,
        "OCV": OCV,
        "tau1": tau1,
        "tau2": tau2
    }


# ============================================================
# 6. PLOT
# ============================================================

def _format_config_label(lambda_value, P0_SCALE):
    return (
        f"$\\lambda={lambda_value}$, "
        f"$P_0={P0_SCALE:.0e}$"
    )


def plot_theta_history(
    time,
    rls_results
):
    """
    Plot dell'evoluzione dei sei coefficienti ARX.

    rls_results ha struttura:

        rls_results[lambda_value][P0_SCALE]
    """

    names = [
        r"$\phi_1$",
        r"$\phi_2$",
        r"$b_0$",
        r"$b_1$",
        r"$b_2$",
        r"$c$"
    ]

    for j, name in enumerate(names):

        plt.figure(figsize=(11, 5))

        for lambda_value, p0_results in rls_results.items():

            for P0_SCALE, result in p0_results.items():

                theta_history = result["theta_history"]
                time_axis = time[2:2 + len(theta_history)]

                plt.plot(
                    time_axis,
                    theta_history[:, j],
                    label=_format_config_label(
                        lambda_value,
                        P0_SCALE
                    ),
                    linewidth=0.8
                )

        plt.xlabel("Time [s]")
        plt.ylabel(name)
        plt.title(f"ARX coefficient {name}")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()


def plot_physical_history(
    physical_results
):
    """
    Plot dei parametri fisici stimati nel tempo.

    physical_results ha struttura:

        physical_results[lambda_value][P0_SCALE]
    """

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("C1", "C1 [F]"),
        ("R2", "R2 [Ohm]"),
        ("C2", "C2 [F]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        plt.figure(figsize=(11, 5))

        plotted_anything = False

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                mask = result["valid"]

                if not np.any(mask):
                    continue

                plotted_anything = True

                plt.plot(
                    result["time"][mask],
                    result[parameter][mask],
                    label=_format_config_label(
                        lambda_value,
                        P0_SCALE
                    ),
                    linewidth=0.8
                )

        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(f"{parameter} evolution")
        plt.grid(True)

        if plotted_anything:
            plt.legend(fontsize=8)

        plt.tight_layout()


def plot_current_and_voltage_and_temperature(
    time,
    voltage,
    current,
    temperature
):
    """
    Visualizza i dati sperimentali.
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7),
        sharex=True
    )

    axes[0].plot(
        time,
        voltage,
        linewidth=0.7
    )

    axes[0].set_ylabel(
        "Voltage [V]"
    )

    axes[0].grid(True)

    axes[1].plot(
        time,
        current,
        linewidth=0.7
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].set_ylabel(
        "Current [A]"
    )

    axes[1].grid(True)


    axes[2].plot(
        time,
        temperature,
        linewidth=0.7
    )

    axes[2].set_ylabel(
        "Cell Temperature [°C]"
    )

    axes[2].set_xlabel(
        "Time [s]"
    )


    axes[2].grid(True)


    plt.tight_layout()

def robust_outlier_mask(
    values,
    valid_mask=None,
    threshold=6.0
):
    """
    Restituisce una maschera booleana che rimuove gli outlier
    usando la Median Absolute Deviation.

    Tiene solo valori:
        - validi secondo valid_mask
        - finiti
        - entro threshold * MAD dalla mediana
    """

    values = np.asarray(values)

    if valid_mask is None:
        mask = np.isfinite(values)
    else:
        mask = valid_mask & np.isfinite(values)

    if np.sum(mask) < 5:
        return mask

    selected = values[mask]

    median = np.median(selected)
    mad = np.median(
        np.abs(selected - median)
    )

    if mad <= 0 or not np.isfinite(mad):
        q1 = np.percentile(selected, 25)
        q3 = np.percentile(selected, 75)
        iqr = q3 - q1

        if iqr <= 0 or not np.isfinite(iqr):
            return mask

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (
            mask
            & (values >= lower)
            & (values <= upper)
        )

    robust_sigma = 1.4826 * mad

    return (
        mask
        & (
            np.abs(values - median)
            <= threshold * robust_sigma
        )
    )


def plot_physical_history_without_outliers(
    physical_results,
    outlier_threshold=6.0
):
    """
    Plot dei parametri fisici stimati rimuovendo gli outlier.
    """

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("C1", "C1 [F]"),
        ("R2", "R2 [Ohm]"),
        ("C2", "C2 [F]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        plt.figure(figsize=(11, 5))

        plotted_anything = False

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                mask = robust_outlier_mask(
                    result[parameter],
                    valid_mask=result["valid"],
                    threshold=outlier_threshold
                )

                if not np.any(mask):
                    continue

                plotted_anything = True

                plt.plot(
                    result["time"][mask],
                    result[parameter][mask],
                    label=_format_config_label(
                        lambda_value,
                        P0_SCALE
                    ),
                    linewidth=0.8
                )

        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(
            f"{parameter} evolution - outlier filtered"
        )
        plt.grid(True)

        if plotted_anything:
            plt.legend(fontsize=8)

        plt.tight_layout()


def plot_physical_history_with_temperature(
    physical_results,
    time,
    temperature,
    outlier_threshold=6.0
):
    """
    Plot dei parametri fisici stimati insieme alla temperatura.

    Asse sinistro:
        parametro fisico filtrato dagli outlier

    Asse destro:
        temperatura cella
    """

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("C1", "C1 [F]"),
        ("R2", "R2 [Ohm]"),
        ("C2", "C2 [F]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        fig, ax_parameter = plt.subplots(
            figsize=(11, 5)
        )

        plotted_anything = False

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                mask = robust_outlier_mask(
                    result[parameter],
                    valid_mask=result["valid"],
                    threshold=outlier_threshold
                )

                if not np.any(mask):
                    continue

                plotted_anything = True

                ax_parameter.plot(
                    result["time"][mask],
                    result[parameter][mask],
                    label=_format_config_label(
                        lambda_value,
                        P0_SCALE
                    ),
                    linewidth=0.8
                )

        ax_parameter.set_xlabel("Time [s]")
        ax_parameter.set_ylabel(ylabel)
        ax_parameter.grid(True)

        ax_temperature = ax_parameter.twinx()

        ax_temperature.plot(
            time,
            temperature,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="Temperature"
        )

        ax_temperature.set_ylabel(
            "Cell Temperature [°C]"
        )

        lines_1, labels_1 = ax_parameter.get_legend_handles_labels()
        lines_2, labels_2 = ax_temperature.get_legend_handles_labels()

        if plotted_anything:
            ax_parameter.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )
        else:
            ax_temperature.legend(
                fontsize=8,
                loc="best"
            )

        plt.title(
            f"{parameter} evolution with temperature"
        )

        fig.tight_layout()


def rolling_median_by_time(
    time,
    values,
    window_seconds
):
    """
    Rolling median centrata su una finestra temporale.
    Serve per rimuovere ripple periodico, non solo outlier.
    """

    time = np.asarray(time)
    values = np.asarray(values)

    y = np.full_like(
        values,
        np.nan,
        dtype=float
    )

    finite = np.isfinite(time) & np.isfinite(values)

    if np.sum(finite) < 5:
        return y

    dt = np.median(
        np.diff(time[finite])
    )

    if not np.isfinite(dt) or dt <= 0:
        return y

    window_samples = int(
        round(window_seconds / dt)
    )

    window_samples = max(
        window_samples,
        5
    )

    if window_samples % 2 == 0:
        window_samples += 1

    smoothed = (
        pd.Series(values)
        .rolling(
            window=window_samples,
            center=True,
            min_periods=max(3, window_samples // 5)
        )
        .median()
        .to_numpy(dtype=float)
    )

    y[finite] = smoothed[finite]

    return y


def compute_charge_state_proxy_ah(
    time,
    current
):
    """
    Proxy dello stato di carica ottenuto integrando la corrente.

    Convenzione corrente del modello:
        current > 0  scarica
        current < 0  carica

    Quindi il proxy aumenta quando la cella si carica.
    """

    time = np.asarray(time)
    current = np.asarray(current)

    dt = np.diff(
        time,
        prepend=time[0]
    )

    charge_state_ah = -np.cumsum(
        current * dt
    ) / 3600.0

    charge_state_ah -= charge_state_ah[0]

    return charge_state_ah


def interpolate_signal(
    source_time,
    source_values,
    target_time
):
    """
    Interpolazione semplice di un segnale sulla griglia fisica.
    """

    return np.interp(
        target_time,
        source_time,
        source_values
    )


def build_current_segments(
    time,
    current,
    step_threshold=CURRENT_STEP_THRESHOLD,
    min_segment_samples=MIN_SEGMENT_SAMPLES
):
    """
    Divide il profilo in segmenti quasi a corrente costante.

    Serve per fare mediane per tratto/ciclo invece che rolling
    median campione per campione.
    """

    time = np.asarray(time)
    current = np.asarray(current)

    current_diff = np.abs(
        np.diff(current, prepend=current[0])
    )

    change_points = np.where(
        current_diff > step_threshold
    )[0]

    starts = np.r_[
        0,
        change_points
    ]

    ends = np.r_[
        change_points,
        len(current)
    ]

    segment_id = np.full(
        len(current),
        -1,
        dtype=int
    )

    valid_segments = []

    segment_counter = 0

    for start, end in zip(starts, ends):

        if end <= start:
            continue

        if end - start < min_segment_samples:
            continue

        segment_id[start:end] = segment_counter

        valid_segments.append(
            {
                "id": segment_counter,
                "start": start,
                "end": end,
                "time": np.median(time[start:end]),
                "current": np.median(current[start:end])
            }
        )

        segment_counter += 1

    return segment_id, valid_segments


def aggregate_physical_by_current_segment(
    result,
    time,
    current,
    temperature,
    charge_proxy_ah,
    parameter,
    voltage=None,
    outlier_threshold=6.0
):
    """
    Aggrega un parametro fisico per segmenti di corrente.

    Ritorna mediane per segmento:
        segment_id
        time
        parameter
        voltage, se fornita
        temperature
        current
        charge_proxy_ah
    """

    segment_id, valid_segments = build_current_segments(
        time,
        current
    )

    physical_time = result["time"]

    physical_segment = np.interp(
        physical_time,
        time,
        segment_id
    ).round().astype(int)

    temp_phys = interpolate_signal(
        time,
        temperature,
        physical_time
    )

    current_phys = interpolate_signal(
        time,
        current,
        physical_time
    )

    charge_phys = interpolate_signal(
        time,
        charge_proxy_ah,
        physical_time
    )

    if voltage is not None:
        voltage_phys = interpolate_signal(
            time,
            voltage,
            physical_time
        )
    else:
        voltage_phys = None

    base_mask = robust_outlier_mask(
        result[parameter],
        valid_mask=result["valid"],
        threshold=outlier_threshold
    )

    rows = []

    for segment in valid_segments:

        sid = segment["id"]

        mask = (
            base_mask
            & (physical_segment == sid)
            & np.isfinite(result[parameter])
            & np.isfinite(temp_phys)
            & np.isfinite(current_phys)
            & np.isfinite(charge_phys)
        )

        if voltage_phys is not None:
            mask = mask & np.isfinite(voltage_phys)

        if np.sum(mask) < 5:
            continue

        row = {
            "segment_id": sid,
            "time": np.median(physical_time[mask]),
            parameter: np.median(result[parameter][mask]),
            "temperature": np.median(temp_phys[mask]),
            "current": np.median(current_phys[mask]),
            "charge_proxy_ah": np.median(charge_phys[mask])
        }

        if voltage_phys is not None:
            row["voltage"] = np.median(voltage_phys[mask])

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def plot_physical_segment_medians_with_temperature(
    physical_results,
    time,
    current,
    temperature,
    outlier_threshold=6.0
):
    """
    Plot dei parametri fisici mediati sui segmenti di corrente.

    Questo riduce molto meglio il ripple rispetto alla rolling median.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("C1", "C1 [F]"),
        ("R2", "R2 [Ohm]"),
        ("C2", "C2 [F]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        fig, ax_parameter = plt.subplots(
            figsize=(11, 5)
        )

        plotted_anything = False

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                df_segment = aggregate_physical_by_current_segment(
                    result,
                    time,
                    current,
                    temperature,
                    charge_proxy_ah,
                    parameter,
                    outlier_threshold=outlier_threshold
                )

                if df_segment.empty:
                    continue

                plotted_anything = True

                ax_parameter.plot(
                    df_segment["time"],
                    df_segment[parameter],
                    marker=".",
                    markersize=3,
                    linewidth=1.0,
                    label=_format_config_label(
                        lambda_value,
                        P0_SCALE
                    )
                )

        ax_temperature = ax_parameter.twinx()

        ax_temperature.plot(
            time,
            temperature,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
            label="Temperature"
        )

        ax_parameter.set_xlabel("Time [s]")
        ax_parameter.set_ylabel(ylabel)
        ax_temperature.set_ylabel("Cell Temperature [°C]")

        ax_parameter.grid(True)

        lines_1, labels_1 = ax_parameter.get_legend_handles_labels()
        lines_2, labels_2 = ax_temperature.get_legend_handles_labels()

        if plotted_anything:
            ax_parameter.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

        plt.title(
            f"{parameter} segment median with temperature"
        )

        fig.tight_layout()

def plot_ocv_vs_ohmic_corrected_voltage(
    physical_results,
    time,
    voltage,
    current,
    temperature,
    outlier_threshold=6.0
):
    """
    Confronta:
        - OCV stimata dal modello;
        - tensione misurata;
        - tensione corretta con +R0*I;
        - tensione corretta con -R0*I.

    Nota:
        V + R0*I corregge solo la caduta ohmica.
        Rimangono i contributi dinamici V_RC1 e V_RC2.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    for lambda_value, p0_results in physical_results.items():

        for P0_SCALE, result in p0_results.items():

            ocv_df = aggregate_physical_by_current_segment(
                result,
                time,
                current,
                temperature,
                charge_proxy_ah,
                "OCV",
                voltage=voltage,
                outlier_threshold=outlier_threshold
            )

            r0_df = aggregate_physical_by_current_segment(
                result,
                time,
                current,
                temperature,
                charge_proxy_ah,
                "R0",
                voltage=voltage,
                outlier_threshold=outlier_threshold
            )

            if ocv_df.empty or r0_df.empty:
                continue

            df = pd.merge(
                ocv_df[
                    [
                        "segment_id",
                        "time",
                        "OCV",
                        "voltage",
                        "current",
                        "temperature"
                    ]
                ],
                r0_df[
                    [
                        "segment_id",
                        "R0"
                    ]
                ],
                on="segment_id",
                how="inner"
            )

            df = df.sort_values(
                "time"
            ).reset_index(drop=True)

            if len(df) < 5:
                continue

            t_plot = df["time"].to_numpy(dtype=float)
            ocv_model = df["OCV"].to_numpy(dtype=float)
            voltage_seg = df["voltage"].to_numpy(dtype=float)
            current_seg = df["current"].to_numpy(dtype=float)
            temp_seg = df["temperature"].to_numpy(dtype=float)
            r0_seg = df["R0"].to_numpy(dtype=float)

            voltage_plus_r0i = (
                voltage_seg
                + r0_seg * current_seg
            )

            voltage_minus_r0i = (
                voltage_seg
                - r0_seg * current_seg
            )

            error_plus = voltage_plus_r0i - ocv_model
            error_minus = voltage_minus_r0i - ocv_model

            rmse_plus = np.sqrt(
                np.nanmean(error_plus ** 2)
            )

            rmse_minus = np.sqrt(
                np.nanmean(error_minus ** 2)
            )

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            print(
                "\nOCV vs ohmic-corrected voltage:"
                f"\n  {label}"
                f"\n  RMSE V + R0*I vs OCV = {rmse_plus:.6g} V"
                f"\n  RMSE V - R0*I vs OCV = {rmse_minus:.6g} V"
            )

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(11, 9),
                sharex=True
            )

            axes[0].plot(
                t_plot,
                ocv_model,
                label="OCV from ARX",
                linewidth=1.5
            )

            axes[0].plot(
                t_plot,
                voltage_seg,
                label="Measured voltage, segment median",
                linewidth=0.8,
                alpha=0.5
            )

            axes[0].plot(
                t_plot,
                voltage_plus_r0i,
                label="Voltage + R0 * I",
                linewidth=1.1
            )

            axes[0].plot(
                t_plot,
                voltage_minus_r0i,
                label="Voltage - R0 * I",
                linewidth=0.8,
                alpha=0.45
            )

            axes[0].set_ylabel("Voltage [V]")
            axes[0].set_title(
                f"OCV vs ohmic-corrected voltage - {label}"
            )
            axes[0].grid(True)
            axes[0].legend(fontsize=8)

            axes[1].plot(
                t_plot,
                error_plus,
                label="V + R0*I - OCV",
                linewidth=1.0
            )

            axes[1].plot(
                t_plot,
                error_minus,
                label="V - R0*I - OCV",
                linewidth=0.8,
                alpha=0.55
            )

            axes[1].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[1].set_ylabel("Difference [V]")
            axes[1].grid(True)
            axes[1].legend(fontsize=8)

            axes[2].plot(
                t_plot,
                current_seg,
                label="Segment current",
                linewidth=0.8
            )

            ax_temp = axes[2].twinx()

            ax_temp.plot(
                t_plot,
                temp_seg,
                color="black",
                linestyle="--",
                linewidth=0.9,
                label="Temperature"
            )

            axes[2].set_xlabel("Time [s]")
            axes[2].set_ylabel("Current [A]")
            ax_temp.set_ylabel("Cell Temperature [°C]")

            axes[2].grid(True)

            lines_1, labels_1 = axes[2].get_legend_handles_labels()
            lines_2, labels_2 = ax_temp.get_legend_handles_labels()

            axes[2].legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

            fig.tight_layout()


def plot_physical_history_smoothed(
    physical_results,
    smooth_window_seconds=SMOOTH_WINDOW_SECONDS,
    outlier_threshold=6.0,
    show_raw=SHOW_RAW_IN_FILTERED_PLOTS
):
    """
    Plot dei parametri fisici:
        - grezzo filtrato dagli outlier, opzionale;
        - rolling median per rimuovere il ripple periodico.
    """

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("C1", "C1 [F]"),
        ("R2", "R2 [Ohm]"),
        ("C2", "C2 [F]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        plt.figure(figsize=(11, 5))

        plotted_anything = False

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                mask = robust_outlier_mask(
                    result[parameter],
                    valid_mask=result["valid"],
                    threshold=outlier_threshold
                )

                if not np.any(mask):
                    continue

                plotted_anything = True

                time_valid = result["time"][mask]
                value_valid = result[parameter][mask]

                value_smooth = rolling_median_by_time(
                    time_valid,
                    value_valid,
                    smooth_window_seconds
                )

                label = _format_config_label(
                    lambda_value,
                    P0_SCALE
                )

                if show_raw:
                    plt.plot(
                        time_valid,
                        value_valid,
                        linewidth=0.4,
                        alpha=0.18,
                        label=f"{label} raw"
                    )

                smooth_mask = np.isfinite(value_smooth)

                plt.plot(
                    time_valid[smooth_mask],
                    value_smooth[smooth_mask],
                    linewidth=1.5,
                    label=f"{label} rolling median"
                )

        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(
            f"{parameter} evolution - smoothed"
        )
        plt.grid(True)

        if plotted_anything:
            plt.legend(fontsize=8)

        plt.tight_layout()


def plot_ocv_diagnostics(
    physical_results,
    time,
    current,
    temperature,
    smooth_window_seconds=SMOOTH_WINDOW_SECONDS,
    outlier_threshold=6.0
):
    """
    Diagnostica OCV.

    Mostra:
        1. OCV stimata, temperatura e proxy SOC;
        2. OCV corretta rispetto alla temperatura;
        3. contributo ohmico R0 * I.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    for lambda_value, p0_results in physical_results.items():

        for P0_SCALE, result in p0_results.items():

            ocv_mask = robust_outlier_mask(
                result["OCV"],
                valid_mask=result["valid"],
                threshold=outlier_threshold
            )

            r0_mask = robust_outlier_mask(
                result["R0"],
                valid_mask=result["valid"],
                threshold=outlier_threshold
            )

            mask = ocv_mask & r0_mask

            if np.sum(mask) < 20:
                continue

            physical_time = result["time"][mask]

            ocv = result["OCV"][mask]
            r0 = result["R0"][mask]

            temp_phys = interpolate_signal(
                time,
                temperature,
                physical_time
            )

            current_phys = interpolate_signal(
                time,
                current,
                physical_time
            )

            charge_phys = interpolate_signal(
                time,
                charge_proxy_ah,
                physical_time
            )

            ocv_smooth = rolling_median_by_time(
                physical_time,
                ocv,
                smooth_window_seconds
            )

            r0_smooth = rolling_median_by_time(
                physical_time,
                r0,
                smooth_window_seconds
            )

            smooth_mask = (
                np.isfinite(ocv_smooth)
                & np.isfinite(r0_smooth)
                & np.isfinite(temp_phys)
                & np.isfinite(charge_phys)
            )

            if np.sum(smooth_mask) < 20:
                continue

            t_plot = physical_time[smooth_mask]
            ocv_plot = ocv_smooth[smooth_mask]
            r0_plot = r0_smooth[smooth_mask]
            temp_plot = temp_phys[smooth_mask]
            charge_plot = charge_phys[smooth_mask]
            current_plot = current_phys[smooth_mask]

            # Regressione lineare:
            # OCV = a + bT * T + bQ * Q
            X = np.column_stack(
                [
                    np.ones_like(temp_plot),
                    temp_plot,
                    charge_plot
                ]
            )

            beta, _, _, _ = np.linalg.lstsq(
                X,
                ocv_plot,
                rcond=None
            )

            temp_ref = np.median(temp_plot)
            charge_ref = charge_plot[0]

            ocv_temp_corrected = (
                ocv_plot
                - beta[1] * (temp_plot - temp_ref)
            )

            ocv_temp_charge_corrected = (
                ocv_plot
                - beta[1] * (temp_plot - temp_ref)
                - beta[2] * (charge_plot - charge_ref)
            )

            ohmic_drop = r0_plot * current_plot

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(11, 9),
                sharex=True
            )

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            axes[0].plot(
                t_plot,
                ocv_plot,
                label="OCV smoothed",
                linewidth=1.4
            )

            ax0_temp = axes[0].twinx()

            ax0_temp.plot(
                t_plot,
                temp_plot,
                color="black",
                linestyle="--",
                linewidth=0.9,
                label="Temperature"
            )

            axes[0].set_ylabel("OCV [V]")
            ax0_temp.set_ylabel("Temperature [°C]")
            axes[0].grid(True)
            axes[0].set_title(
                f"OCV diagnostic - {label}"
            )

            lines_1, labels_1 = axes[0].get_legend_handles_labels()
            lines_2, labels_2 = ax0_temp.get_legend_handles_labels()

            axes[0].legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

            axes[1].plot(
                t_plot,
                ocv_plot,
                label="OCV smoothed",
                linewidth=1.0,
                alpha=0.6
            )

            axes[1].plot(
                t_plot,
                ocv_temp_corrected,
                label="OCV corrected for temperature",
                linewidth=1.4
            )

            axes[1].plot(
                t_plot,
                ocv_temp_charge_corrected,
                label="OCV corrected for temperature + charge proxy",
                linewidth=1.4
            )

            axes[1].set_ylabel("OCV [V]")
            axes[1].grid(True)
            axes[1].legend(fontsize=8)

            axes[2].plot(
                t_plot,
                r0_plot,
                label="R0 smoothed",
                linewidth=1.3
            )

            ax2_drop = axes[2].twinx()

            ax2_drop.plot(
                t_plot,
                ohmic_drop,
                color="tab:red",
                linewidth=0.8,
                alpha=0.7,
                label="R0 * I"
            )

            axes[2].set_xlabel("Time [s]")
            axes[2].set_ylabel("R0 [Ohm]")
            ax2_drop.set_ylabel("Ohmic contribution [V]")
            axes[2].grid(True)

            lines_1, labels_1 = axes[2].get_legend_handles_labels()
            lines_2, labels_2 = ax2_drop.get_legend_handles_labels()

            axes[2].legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

            print(
                "\nOCV diagnostic:"
                f"\n  lambda = {lambda_value}, P0_SCALE = {P0_SCALE:.1e}"
                f"\n  dOCV/dT ≈ {beta[1]:.6g} V/°C"
                f"\n  dOCV/dQ ≈ {beta[2]:.6g} V/Ah"
            )

            fig.tight_layout()


def plot_ocv_pair_averaged_voltage(
    physical_results,
    time,
    voltage,
    current,
    temperature,
    outlier_threshold=6.0
):
    """
    Confronta OCV con la tensione ohmic-corrected mediata tra
    segmenti consecutivi a corrente opposta.

    Questo aiuta a ridurre il contributo alternato della corrente.
    Non elimina completamente le dinamiche RC.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    for lambda_value, p0_results in physical_results.items():

        for P0_SCALE, result in p0_results.items():

            ocv_df = aggregate_physical_by_current_segment(
                result,
                time,
                current,
                temperature,
                charge_proxy_ah,
                "OCV",
                voltage=voltage,
                outlier_threshold=outlier_threshold
            )

            r0_df = aggregate_physical_by_current_segment(
                result,
                time,
                current,
                temperature,
                charge_proxy_ah,
                "R0",
                voltage=voltage,
                outlier_threshold=outlier_threshold
            )

            if ocv_df.empty or r0_df.empty:
                continue

            df = pd.merge(
                ocv_df[
                    [
                        "segment_id",
                        "time",
                        "OCV",
                        "voltage",
                        "current",
                        "temperature"
                    ]
                ],
                r0_df[
                    [
                        "segment_id",
                        "R0"
                    ]
                ],
                on="segment_id",
                how="inner"
            )

            df = df.sort_values(
                "time"
            ).reset_index(drop=True)

            df["voltage_plus_r0i"] = (
                df["voltage"]
                + df["R0"] * df["current"]
            )

            rows = []

            for idx in range(len(df) - 1):

                i1 = df.loc[idx, "current"]
                i2 = df.loc[idx + 1, "current"]

                if i1 == 0 or i2 == 0:
                    continue

                if np.sign(i1) == np.sign(i2):
                    continue

                rows.append(
                    {
                        "time": 0.5 * (
                            df.loc[idx, "time"]
                            + df.loc[idx + 1, "time"]
                        ),
                        "OCV": 0.5 * (
                            df.loc[idx, "OCV"]
                            + df.loc[idx + 1, "OCV"]
                        ),
                        "voltage_plus_r0i_pair": 0.5 * (
                            df.loc[idx, "voltage_plus_r0i"]
                            + df.loc[idx + 1, "voltage_plus_r0i"]
                        ),
                        "temperature": 0.5 * (
                            df.loc[idx, "temperature"]
                            + df.loc[idx + 1, "temperature"]
                        )
                    }
                )

            if not rows:
                continue

            pair_df = pd.DataFrame(rows)

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            difference = (
                pair_df["OCV"]
                - pair_df["voltage_plus_r0i_pair"]
            )

            mean_difference = np.nanmean(difference)
            std_difference = np.nanstd(difference)

            print(
                "\nPair-averaged OCV comparison:"
                f"\n  {label}"
                f"\n  mean OCV - proxy = {mean_difference:.6g} V"
                f"\n  std  OCV - proxy = {std_difference:.6g} V"
            )

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7),
                sharex=True,
                gridspec_kw={
                    "height_ratios": [2, 1]
                }
            )

            axes[0].plot(
                pair_df["time"],
                pair_df["OCV"],
                label="OCV from ARX, pair average",
                linewidth=1.5
            )

            axes[0].plot(
                pair_df["time"],
                pair_df["voltage_plus_r0i_pair"],
                label="Pair average of V + R0*I",
                linewidth=1.2
            )

            axes[0].set_ylabel("Voltage [V]")
            axes[0].grid(True)

            ax_temp = axes[0].twinx()

            ax_temp.plot(
                pair_df["time"],
                pair_df["temperature"],
                color="black",
                linestyle="--",
                linewidth=0.9,
                label="Temperature"
            )

            ax_temp.set_ylabel("Cell Temperature [°C]")

            lines_1, labels_1 = axes[0].get_legend_handles_labels()
            lines_2, labels_2 = ax_temp.get_legend_handles_labels()

            axes[0].legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

            axes[0].set_title(
                f"OCV vs pair-averaged ohmic-corrected voltage - {label}"
            )

            axes[1].plot(
                pair_df["time"],
                difference,
                label="OCV - pair average of V + R0*I",
                linewidth=1.2
            )

            axes[1].axhline(
                mean_difference,
                color="tab:red",
                linestyle="--",
                linewidth=0.9,
                label=f"mean = {mean_difference:.4f} V"
            )

            axes[1].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Difference [V]")
            axes[1].grid(True)
            axes[1].legend(fontsize=8)

            fig.tight_layout()


def downsample_indices(
    n,
    max_points=20000
):
    """
    Indici per scatter plot leggeri.
    """

    if n <= max_points:
        return np.arange(n)

    return np.linspace(
        0,
        n - 1,
        max_points
    ).astype(int)


def rolling_rms_by_time(
    time,
    values,
    window_seconds
):
    """
    RMS mobile centrato su finestra temporale.
    """

    time = np.asarray(time)
    values = np.asarray(values)

    y = np.full_like(
        values,
        np.nan,
        dtype=float
    )

    finite = np.isfinite(time) & np.isfinite(values)

    if np.sum(finite) < 5:
        return y

    dt = np.median(
        np.diff(time[finite])
    )

    if not np.isfinite(dt) or dt <= 0:
        return y

    window_samples = int(
        round(window_seconds / dt)
    )

    window_samples = max(
        window_samples,
        5
    )

    if window_samples % 2 == 0:
        window_samples += 1

    rms = np.sqrt(
        pd.Series(values ** 2)
        .rolling(
            window=window_samples,
            center=True,
            min_periods=max(3, window_samples // 5)
        )
        .mean()
        .to_numpy(dtype=float)
    )

    y[finite] = rms[finite]

    return y


def plot_rls_residual_diagnostics(
    time,
    current,
    temperature,
    rls_results,
    smooth_window_seconds=SMOOTH_WINDOW_SECONDS
):
    """
    Diagnostica dei residui RLS.

    Da guardare:
        - errore medio vicino a zero;
        - RMS basso e non crescente;
        - nessuna dipendenza chiara da corrente;
        - nessuna dipendenza chiara da temperatura.
    """

    for lambda_value, p0_results in rls_results.items():

        for P0_SCALE, result in p0_results.items():

            error = result["error_history"]

            t_error = time[
                2:2 + len(error)
            ]

            current_error = current[
                2:2 + len(error)
            ]

            temp_error = temperature[
                2:2 + len(error)
            ]

            error_rms = rolling_rms_by_time(
                t_error,
                error,
                smooth_window_seconds
            )

            idx = downsample_indices(
                len(error)
            )

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(11, 10)
            )

            axes[0].plot(
                t_error,
                error,
                linewidth=0.35,
                alpha=0.5
            )

            axes[0].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[0].set_ylabel("Error [V]")
            axes[0].set_title(
                f"RLS residual diagnostics - {label}"
            )
            axes[0].grid(True)

            axes[1].plot(
                t_error,
                error_rms,
                linewidth=1.2
            )

            axes[1].set_ylabel("Rolling RMS [V]")
            axes[1].grid(True)

            axes[2].scatter(
                current_error[idx],
                error[idx],
                s=2,
                alpha=0.25
            )

            axes[2].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[2].set_xlabel("Current [A]")
            axes[2].set_ylabel("Error [V]")
            axes[2].grid(True)

            axes[3].scatter(
                temp_error[idx],
                error[idx],
                s=2,
                alpha=0.25
            )

            axes[3].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[3].set_xlabel("Temperature [°C]")
            axes[3].set_ylabel("Error [V]")
            axes[3].grid(True)

            fig.tight_layout()


def plot_segment_parameter_vs_temperature(
    physical_results,
    time,
    current,
    temperature,
    outlier_threshold=6.0
):
    """
    Scatter dei parametri fisici mediati per segmento contro temperatura.

    Colore = corrente del segmento.

    Serve per capire se il parametro dipende davvero da T oppure
    se sta assorbendo il ciclo di corrente.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    parameters = [
        ("R0", "R0 [Ohm]"),
        ("R1", "R1 [Ohm]"),
        ("R2", "R2 [Ohm]"),
        ("tau1", "tau1 [s]"),
        ("tau2", "tau2 [s]"),
        ("OCV", "OCV [V]")
    ]

    for parameter, ylabel in parameters:

        for lambda_value, p0_results in physical_results.items():

            for P0_SCALE, result in p0_results.items():

                df_segment = aggregate_physical_by_current_segment(
                    result,
                    time,
                    current,
                    temperature,
                    charge_proxy_ah,
                    parameter,
                    outlier_threshold=outlier_threshold
                )

                if df_segment.empty:
                    continue

                label = _format_config_label(
                    lambda_value,
                    P0_SCALE
                )

                plt.figure(
                    figsize=(8, 6)
                )

                scatter = plt.scatter(
                    df_segment["temperature"],
                    df_segment[parameter],
                    c=df_segment["current"],
                    s=8,
                    alpha=0.75,
                    cmap="coolwarm"
                )

                plt.xlabel("Cell Temperature [°C]")
                plt.ylabel(ylabel)
                plt.title(
                    f"{parameter} vs temperature - {label}"
                )
                plt.grid(True)

                cbar = plt.colorbar(scatter)
                cbar.set_label("Segment current [A]")

                plt.tight_layout()

def plot_ocv_vs_charge_proxy_and_temperature(
    physical_results,
    time,
    current,
    temperature,
    outlier_threshold=6.0
):
    """
    Scatter OCV vs proxy di carica.

    Colore = temperatura.

    Se l'OCV fosse principalmente legata al SOC, dovrebbe esserci
    una curva abbastanza ordinata rispetto a charge_proxy_ah.
    Se domina la temperatura, il colore organizza il grafico.
    """

    charge_proxy_ah = compute_charge_state_proxy_ah(
        time,
        current
    )

    for lambda_value, p0_results in physical_results.items():

        for P0_SCALE, result in p0_results.items():

            df_segment = aggregate_physical_by_current_segment(
                result,
                time,
                current,
                temperature,
                charge_proxy_ah,
                "OCV",
                outlier_threshold=outlier_threshold
            )

            if df_segment.empty:
                continue

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5)
            )

            scatter_1 = axes[0].scatter(
                df_segment["charge_proxy_ah"],
                df_segment["OCV"],
                c=df_segment["temperature"],
                s=8,
                alpha=0.75,
                cmap="viridis"
            )

            axes[0].set_xlabel("Charge proxy [Ah]")
            axes[0].set_ylabel("OCV [V]")
            axes[0].set_title("OCV vs charge proxy")
            axes[0].grid(True)

            cbar_1 = fig.colorbar(
                scatter_1,
                ax=axes[0]
            )
            cbar_1.set_label("Cell Temperature [°C]")

            scatter_2 = axes[1].scatter(
                df_segment["temperature"],
                df_segment["OCV"],
                c=df_segment["charge_proxy_ah"],
                s=8,
                alpha=0.75,
                cmap="plasma"
            )

            axes[1].set_xlabel("Cell Temperature [°C]")
            axes[1].set_ylabel("OCV [V]")
            axes[1].set_title("OCV vs temperature")
            axes[1].grid(True)

            cbar_2 = fig.colorbar(
                scatter_2,
                ax=axes[1]
            )
            cbar_2.set_label("Charge proxy [Ah]")

            fig.suptitle(
                f"OCV dependency map - {label}"
            )

            fig.tight_layout()


def plot_arx_poles_history(
    time,
    rls_results
):
    """
    Plot dei poli discreti ARX nel tempo.

    Per un modello fisico 2RC ci si aspetta:
        0 < alpha2 < alpha1 < 1
    """

    for lambda_value, p0_results in rls_results.items():

        for P0_SCALE, result in p0_results.items():

            theta_history = result["theta_history"]

            alpha1 = np.full(
                len(theta_history),
                np.nan
            )

            alpha2 = np.full(
                len(theta_history),
                np.nan
            )

            for idx, theta in enumerate(theta_history):

                phi1 = theta[0]
                phi2 = theta[1]

                roots = np.roots(
                    [
                        1.0,
                        -phi1,
                        -phi2
                    ]
                )

                if np.any(
                    np.abs(roots.imag) > 1e-8
                ):
                    continue

                roots = np.sort(
                    roots.real
                )[::-1]

                alpha1[idx] = roots[0]
                alpha2[idx] = roots[1]

            t_plot = time[
                2:2 + len(theta_history)
            ]

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(11, 7),
                sharex=True
            )

            axes[0].plot(
                t_plot,
                alpha1,
                label=r"$\alpha_1$",
                linewidth=0.8
            )

            axes[0].plot(
                t_plot,
                alpha2,
                label=r"$\alpha_2$",
                linewidth=0.8
            )

            axes[0].axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=0.8
            )

            axes[0].axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=0.8
            )

            axes[0].set_ylabel("Discrete pole")
            axes[0].set_title(
                f"ARX poles history - {label}"
            )
            axes[0].grid(True)
            axes[0].legend(fontsize=8)

            valid_region = (
                np.isfinite(alpha1)
                & np.isfinite(alpha2)
                & (alpha1 < 1.0)
                & (alpha1 > 0.0)
                & (alpha2 < alpha1)
                & (alpha2 > 0.0)
            )

            axes[1].plot(
                t_plot,
                valid_region.astype(float),
                linewidth=0.8
            )

            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Physical pole flag")
            axes[1].set_yticks([0, 1])
            axes[1].grid(True)

            fig.tight_layout()


# ============================================================
# 6B. VALIDAZIONE DIGITAL TWIN
# ============================================================

def compute_error_metrics(
    error
):
    """
    Calcola metriche base su un vettore errore.
    """

    error = np.asarray(
        error,
        dtype=float
    )

    mask = np.isfinite(error)

    if not np.any(mask):
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "max_abs": np.nan,
            "p95_abs": np.nan
        }

    e = error[mask]

    return {
        "n": len(e),
        "bias": np.mean(e),
        "rmse": np.sqrt(np.mean(e ** 2)),
        "mae": np.mean(np.abs(e)),
        "max_abs": np.max(np.abs(e)),
        "p95_abs": np.percentile(np.abs(e), 95)
    }


def print_metric_row(
    name,
    metrics
):
    """
    Stampa una riga di metriche.
    """

    print(
        f"  {name:<24}"
        f" n={metrics['n']:>8d}"
        f" | bias={metrics['bias'] * 1000:>9.3f} mV"
        f" | RMSE={metrics['rmse'] * 1000:>9.3f} mV"
        f" | MAE={metrics['mae'] * 1000:>9.3f} mV"
        f" | P95={metrics['p95_abs'] * 1000:>9.3f} mV"
        f" | MAX={metrics['max_abs'] * 1000:>9.3f} mV"
    )



def simulate_free_run_adaptive(
    voltage,
    current,
    theta_history,
    start_index=2,
    divergence_limit=10.0
):
    """
    Simulazione libera adaptive.

    Usa theta_history online, ma per i regressori usa tensioni
    simulate, non tensioni misurate.

    La simulazione parte da start_index, non necessariamente da zero.

    Inizializzazione:
        V_hat[start_index - 2] = V_measured[start_index - 2]
        V_hat[start_index - 1] = V_measured[start_index - 1]

    Modello:
        V[k] = phi1 V[k-1] + phi2 V[k-2]
             + b0 I[k] + b1 I[k-1] + b2 I[k-2] + c
    """

    voltage = np.asarray(
        voltage,
        dtype=float
    )

    current = np.asarray(
        current,
        dtype=float
    )

    voltage_hat = np.full_like(
        voltage,
        np.nan,
        dtype=float
    )

    if len(voltage) < 3:
        return voltage_hat

    start_index = int(start_index)

    start_index = max(
        start_index,
        2
    )

    start_index = min(
        start_index,
        len(voltage) - 1
    )

    # Inizializzazione locale con valori misurati.
    voltage_hat[start_index - 2] = voltage[start_index - 2]
    voltage_hat[start_index - 1] = voltage[start_index - 1]

    for k in range(start_index, len(voltage)):

        theta_idx = k - 2

        if theta_idx < 0 or theta_idx >= len(theta_history):
            break

        theta = theta_history[theta_idx]

        phi1, phi2, b0, b1, b2, c = theta

        voltage_hat[k] = (
            phi1 * voltage_hat[k - 1]
            + phi2 * voltage_hat[k - 2]
            + b0 * current[k]
            + b1 * current[k - 1]
            + b2 * current[k - 2]
            + c
        )

        # Protezione da eventuali divergenze numeriche.
        if (
            not np.isfinite(voltage_hat[k])
            or abs(voltage_hat[k]) > divergence_limit
        ):
            voltage_hat[k:] = np.nan
            break

    return voltage_hat


def time_to_index(
    time,
    target_time
):
    """
    Converte un tempo in indice campione.
    """

    return int(
        np.searchsorted(
            time,
            target_time
        )
    )


def build_validation_mask(
    time,
    warmup_seconds=VALIDATION_WARMUP_SECONDS,
    end_trim_seconds=VALIDATION_END_TRIM_SECONDS
):
    """
    Maschera di validazione che esclude warm-up e coda finale.
    """

    mask = time >= warmup_seconds

    if end_trim_seconds > 0:
        mask = mask & (
            time <= time[-1] - end_trim_seconds
        )

    return mask


def print_dt_validation_metrics(
    time,
    voltage,
    current,
    temperature,
    rls_results
):
    """
    Metriche quantitative per validazione Digital Twin.

    Confronta:
        - one-step prediction;
        - free-run adaptive simulation.
    """

    print("\n" + "=" * 80)
    print("DIGITAL TWIN VALIDATION METRICS")
    print("=" * 80)

    for lambda_value, p0_results in rls_results.items():

        for P0_SCALE, result in p0_results.items():

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            theta_history = result["theta_history"]
            error_one_step = result["error_history"]

            t_eval = time[
                2:2 + len(error_one_step)
            ]

            voltage_eval = voltage[
                2:2 + len(error_one_step)
            ]

            current_eval = current[
                2:2 + len(error_one_step)
            ]

            temp_eval = temperature[
                2:2 + len(error_one_step)
            ]

            # Se error = V_measured - V_predicted:
            voltage_one_step = (
                voltage_eval
                - error_one_step
            )

            free_run_start_index = time_to_index(
                time,
                VALIDATION_WARMUP_SECONDS
            )

            free_run_voltage = simulate_free_run_adaptive(
                voltage,
                current,
                theta_history,
                start_index=free_run_start_index
            )


            free_run_eval = free_run_voltage[
                2:2 + len(error_one_step)
            ]

            error_free_run = (
                voltage_eval
                - free_run_eval
            )

            validation_mask = build_validation_mask(
                t_eval
            )

            finite_mask = (
                validation_mask
                & np.isfinite(error_one_step)
                & np.isfinite(error_free_run)
                & np.isfinite(current_eval)
                & np.isfinite(temp_eval)
            )

            if not np.any(finite_mask):
                continue

            subsets = {
                "all": finite_mask,
                "charge I<0": (
                    finite_mask
                    & (
                        current_eval
                        < -CURRENT_SIGN_THRESHOLD
                    )
                ),
                "discharge I>0": (
                    finite_mask
                    & (
                        current_eval
                        > CURRENT_SIGN_THRESHOLD
                    )
                ),
                "low T": (
                    finite_mask
                    & (
                        temp_eval
                        < LOW_TEMPERATURE_THRESHOLD
                    )
                ),
                "mid T": (
                    finite_mask
                    & (
                        temp_eval
                        >= LOW_TEMPERATURE_THRESHOLD
                    )
                    & (
                        temp_eval
                        <= HIGH_TEMPERATURE_THRESHOLD
                    )
                ),
                "high T": (
                    finite_mask
                    & (
                        temp_eval
                        > HIGH_TEMPERATURE_THRESHOLD
                    )
                )
            }

            print(
                f"\nConfiguration: {label}"
            )

            print("\nOne-step prediction error:")
            for subset_name, subset_mask in subsets.items():
                print_metric_row(
                    subset_name,
                    compute_error_metrics(
                        error_one_step[subset_mask]
                    )
                )

            print("\nFree-run adaptive simulation error:")
            for subset_name, subset_mask in subsets.items():
                print_metric_row(
                    subset_name,
                    compute_error_metrics(
                        error_free_run[subset_mask]
                    )
                )


def plot_dt_validation(
    time,
    voltage,
    current,
    temperature,
    rls_results,
    rms_window_seconds=VALIDATION_RMS_WINDOW_SECONDS
):
    """
    Plot di validazione DT:
        - tensione misurata vs one-step vs free-run;
        - errore one-step;
        - errore free-run;
        - rolling RMS;
        - corrente e temperatura.
    """

    for lambda_value, p0_results in rls_results.items():

        for P0_SCALE, result in p0_results.items():

            theta_history = result["theta_history"]
            error_one_step = result["error_history"]

            t_eval = time[
                2:2 + len(error_one_step)
            ]

            voltage_eval = voltage[
                2:2 + len(error_one_step)
            ]

            current_eval = current[
                2:2 + len(error_one_step)
            ]

            temp_eval = temperature[
                2:2 + len(error_one_step)
            ]

            voltage_one_step = (
                voltage_eval
                - error_one_step
            )

            free_run_start_index = time_to_index(
                time,
                VALIDATION_WARMUP_SECONDS
            )

            free_run_voltage = simulate_free_run_adaptive(
                voltage,
                current,
                theta_history,
                start_index=free_run_start_index
            )

            free_run_eval = free_run_voltage[
                2:2 + len(error_one_step)
            ]

            error_free_run = (
                voltage_eval
                - free_run_eval
            )

            validation_mask = build_validation_mask(
                t_eval
            )

            t_plot = t_eval[validation_mask]
            voltage_plot = voltage_eval[validation_mask]
            one_step_plot = voltage_one_step[validation_mask]
            free_run_plot = free_run_eval[validation_mask]

            error_one_plot = error_one_step[validation_mask]
            error_free_plot = error_free_run[validation_mask]

            current_plot = current_eval[validation_mask]
            temp_plot = temp_eval[validation_mask]

            rms_one = rolling_rms_by_time(
                t_plot,
                error_one_plot,
                rms_window_seconds
            )

            rms_free = rolling_rms_by_time(
                t_plot,
                error_free_plot,
                rms_window_seconds
            )

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            fig, axes = plt.subplots(
                5,
                1,
                figsize=(12, 12),
                sharex=True
            )

            axes[0].plot(
                t_plot,
                voltage_plot,
                label="Measured voltage",
                linewidth=2.5,
            )

            axes[0].plot(
                t_plot,
                one_step_plot,
                label="One-step prediction",
                linewidth=1.5,
                alpha=0.8,
                linestyle="--",
            )

            axes[0].plot(
                t_plot,
                free_run_plot,
                label="Free-run adaptive",
                linewidth=0.8,
                alpha=0.8
            )

            axes[0].set_ylabel("Voltage [V]")
            axes[0].set_title(
                f"Digital Twin validation - {label}"
            )
            axes[0].grid(True)
            axes[0].legend(fontsize=8)

            axes[1].plot(
                t_plot,
                error_one_plot * 1000,
                linewidth=0.5
            )

            axes[1].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[1].set_ylabel("One-step error [mV]")
            axes[1].grid(True)

            axes[2].plot(
                t_plot,
                error_free_plot * 1000,
                linewidth=0.5
            )

            axes[2].axhline(
                0.0,
                color="black",
                linewidth=0.8
            )

            axes[2].set_ylabel("Free-run error [mV]")
            axes[2].grid(True)

            axes[3].plot(
                t_plot,
                rms_one * 1000,
                label="One-step RMS",
                linewidth=1.2
            )

            axes[3].plot(
                t_plot,
                rms_free * 1000,
                label="Free-run RMS",
                linewidth=1.2
            )

            axes[3].set_ylabel("Rolling RMS [mV]")
            axes[3].grid(True)
            axes[3].legend(fontsize=8)

            axes[4].plot(
                t_plot,
                current_plot,
                label="Current",
                linewidth=0.7
            )

            ax_temp = axes[4].twinx()

            ax_temp.plot(
                t_plot,
                temp_plot,
                color="black",
                linestyle="--",
                linewidth=0.8,
                label="Temperature"
            )

            axes[4].set_xlabel("Time [s]")
            axes[4].set_ylabel("Current [A]")
            ax_temp.set_ylabel("Temperature [°C]")
            axes[4].grid(True)

            lines_1, labels_1 = axes[4].get_legend_handles_labels()
            lines_2, labels_2 = ax_temp.get_legend_handles_labels()

            axes[4].legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                fontsize=8,
                loc="best"
            )

            fig.tight_layout()


def plot_dt_validation_zoom_windows(
    time,
    voltage,
    current,
    temperature,
    rls_results,
    window_centers=None,
    window_seconds=12000
):
    """
    Zoom su finestre temporali per vedere se l'errore nasce
    sui fronti di corrente o sui plateau.

    Se window_centers=None usa tre finestre:
        alta T, media T, bassa T.
    """

    if window_centers is None:
        window_centers = [
            250000,
            500000,
            750000
        ]

    for lambda_value, p0_results in rls_results.items():

        for P0_SCALE, result in p0_results.items():

            theta_history = result["theta_history"]
            error_one_step = result["error_history"]

            t_eval = time[
                2:2 + len(error_one_step)
            ]

            voltage_eval = voltage[
                2:2 + len(error_one_step)
            ]

            current_eval = current[
                2:2 + len(error_one_step)
            ]

            temp_eval = temperature[
                2:2 + len(error_one_step)
            ]

            voltage_one_step = (
                voltage_eval
                - error_one_step
            )

            label = _format_config_label(
                lambda_value,
                P0_SCALE
            )

            for center in window_centers:

                window_start_time = (
                    center
                    - window_seconds / 2
                )

                free_run_start_index = time_to_index(
                    time,
                    window_start_time
                )

                free_run_voltage = simulate_free_run_adaptive(
                    voltage,
                    current,
                    theta_history,
                    start_index=free_run_start_index
                )

                free_run_eval = free_run_voltage[
                    2:2 + len(error_one_step)
                ]

                error_free_run = (
                    voltage_eval
                    - free_run_eval
                )
                mask = (
                    t_eval >= center - window_seconds / 2
                ) & (
                    t_eval <= center + window_seconds / 2
                )

                if np.sum(mask) < 10:
                    continue

                fig, axes = plt.subplots(
                    4,
                    1,
                    figsize=(12, 9),
                    sharex=True
                )

                axes[0].plot(
                    t_eval[mask],
                    voltage_eval[mask],
                    label="Measured",
                    linewidth=2.5,
                )

                axes[0].plot(
                    t_eval[mask],
                    voltage_one_step[mask],
                    label="One-step",
                    linewidth=1.5,
                    linestyle="--",
                )

                axes[0].plot(
                    t_eval[mask],
                    free_run_eval[mask],
                    label="Free-run",
                    linewidth=0.8
                )

                axes[0].set_ylabel("Voltage [V]")
                axes[0].set_title(
                    f"DT validation zoom - {label}, center={center:.0f}s"
                )
                axes[0].grid(True)
                axes[0].legend(fontsize=8)

                axes[1].plot(
                    t_eval[mask],
                    current_eval[mask],
                    linewidth=0.8
                )

                axes[1].set_ylabel("Current [A]")
                axes[1].grid(True)

                axes[2].plot(
                    t_eval[mask],
                    error_one_step[mask] * 1000,
                    label="One-step error",
                    linewidth=0.8
                )

                axes[2].plot(
                    t_eval[mask],
                    error_free_run[mask] * 1000,
                    label="Free-run error",
                    linewidth=0.8
                )

                axes[2].axhline(
                    0.0,
                    color="black",
                    linewidth=0.8
                )

                axes[2].set_ylabel("Error [mV]")
                axes[2].grid(True)
                axes[2].legend(fontsize=8)

                axes[3].plot(
                    t_eval[mask],
                    temp_eval[mask],
                    color="black",
                    linewidth=0.9
                )

                axes[3].set_xlabel("Time [s]")
                axes[3].set_ylabel("Temperature [°C]")
                axes[3].grid(True)

                fig.tight_layout()

# ============================================================
# 7. STATISTICHE FINALI
# ============================================================

def print_summary(
    rls_results,
    physical_results
):
    """
    Riassume il comportamento finale e la percentuale di
    campioni fisicamente validi.

    Le strutture sono:

        rls_results[lambda_value][P0_SCALE]
        physical_results[lambda_value][P0_SCALE]
    """

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for lambda_value in LAMBDA_VALUES:

        for P0_SCALE in P0_SCALE_VALUES:

            rls_result = rls_results[
                lambda_value
            ][
                P0_SCALE
            ]

            physical = physical_results[
                lambda_value
            ][
                P0_SCALE
            ]

            valid_fraction = np.mean(
                physical["valid"]
            )

            print(
                f"\nlambda = {lambda_value}, "
                f"P0_SCALE = {P0_SCALE:.1e}"
            )

            print(
                f"  valid physical fraction = "
                f"{100 * valid_fraction:.2f}%"
            )

            print(
                "  final theta =",
                rls_result["theta_final"]
            )

            mask = physical["valid"]

            if np.any(mask):

                last = np.where(mask)[0][-1]

                print(
                    f"  final valid R0   = "
                    f"{physical['R0'][last]:.8g}"
                )

                print(
                    f"  final valid R1   = "
                    f"{physical['R1'][last]:.8g}"
                )

                print(
                    f"  final valid C1   = "
                    f"{physical['C1'][last]:.8g}"
                )

                print(
                    f"  final valid R2   = "
                    f"{physical['R2'][last]:.8g}"
                )

                print(
                    f"  final valid C2   = "
                    f"{physical['C2'][last]:.8g}"
                )

                print(
                    f"  final valid tau1 = "
                    f"{physical['tau1'][last]:.8g}"
                )

                print(
                    f"  final valid tau2 = "
                    f"{physical['tau2'][last]:.8g}"
                )

                print(
                    f"  final valid OCV  = "
                    f"{physical['OCV'][last]:.8g}"
                )

            else:

                print(
                    "  no physically valid samples"
                )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Caricamento
    # --------------------------------------------------------

    time, voltage, current, temperature = load_experiment_csv(
        CSV_FILE
    )

    # Controllo sampling
    check_sampling(time)

    print("\nDataset:")
    print(
        f"  samples = {len(voltage)}"
    )

    print(
        f"  duration = {time[-1]:.2f} s"
    )

    print(
        f"  voltage range = "
        f"{voltage.min():.4f} -> "
        f"{voltage.max():.4f} V"
    )

    print(
        f"  current range = "
        f"{current.min():.4f} -> "
        f"{current.max():.4f} A"
    )

    print(
        f"  temperature range = "
        f"{temperature.min():.4f} -> "
        f"{temperature.max():.4f} °C"
    )


    # --------------------------------------------------------
    # Dati sperimentali
    # --------------------------------------------------------

    plot_current_and_voltage_and_temperature(
        time,
        voltage,
        current,
        temperature
    )

    # --------------------------------------------------------
    # RLS per i diversi lambda e P0
    # --------------------------------------------------------

    rls_results = {}

    for lambda_value in LAMBDA_VALUES:

        rls_results[lambda_value] = {}

        for P0_SCALE in P0_SCALE_VALUES:

            print(
                "\nRunning RLS with "
                f"lambda = {lambda_value}, "
                f"P0_SCALE = {P0_SCALE:.1e} ..."
            )

            rls_results[lambda_value][P0_SCALE] = run_rls(
                voltage,
                current,
                lambda_value,
                P0_SCALE=P0_SCALE
            )

    # --------------------------------------------------------
    # Conversione ARX -> parametri fisici
    # --------------------------------------------------------

    physical_results = {}

    for lambda_value in LAMBDA_VALUES:

        physical_results[lambda_value] = {}

        for P0_SCALE in P0_SCALE_VALUES:

            print(
                "\nConverting physical parameters for "
                f"lambda = {lambda_value}, "
                f"P0_SCALE = {P0_SCALE:.1e} ..."
            )

            physical_results[lambda_value][P0_SCALE] = (
                convert_history_to_physical(
                    rls_results[lambda_value][P0_SCALE][
                        "theta_history"
                    ],
                    time,
                    Ts,
                    PHYSICAL_DECIMATION
                )
            )

    # --------------------------------------------------------
    # Grafici ARX
    # --------------------------------------------------------

    plot_theta_history(
        time,
        rls_results
    )

    # --------------------------------------------------------
    # Grafici parametri fisici
    # --------------------------------------------------------

    plot_physical_history(
        physical_results
    )



    # --------------------------------------------------------
    # Grafici parametri fisici smussati
    # --------------------------------------------------------

    plot_physical_history_smoothed(
        physical_results,
        smooth_window_seconds=SMOOTH_WINDOW_SECONDS,
        outlier_threshold=6.0,
        show_raw=SHOW_RAW_IN_FILTERED_PLOTS
    )

    # --------------------------------------------------------
    # Grafici parametri fisici con temperatura
    # --------------------------------------------------------

    plot_physical_history_with_temperature(
        physical_results,
        time,
        temperature,
        outlier_threshold=6.0
    )

    # --------------------------------------------------------
    # Diagnostica OCV / temperatura / contributo resistivo
    # --------------------------------------------------------

    plot_ocv_diagnostics(
        physical_results,
        time,
        current,
        temperature,
        smooth_window_seconds=SMOOTH_WINDOW_SECONDS,
        outlier_threshold=6.0
    )


    # --------------------------------------------------------
    # Parametri fisici mediati per segmenti di corrente
    # --------------------------------------------------------

    plot_physical_segment_medians_with_temperature(
        physical_results,
        time,
        current,
        temperature,
        outlier_threshold=6.0
    )

    # --------------------------------------------------------
    # OCV vs tensione corretta ohmicamente
    # --------------------------------------------------------

    plot_ocv_vs_ohmic_corrected_voltage(
        physical_results,
        time,
        voltage,
        current,
        temperature,
        outlier_threshold=6.0
    )


    plot_ocv_pair_averaged_voltage(
        physical_results,
        time,
        voltage,
        current,
        temperature,
        outlier_threshold=6.0
    )


    # --------------------------------------------------------
    # Diagnostica residui RLS
    # --------------------------------------------------------

    plot_rls_residual_diagnostics(
        time,
        current,
        temperature,
        rls_results,
        smooth_window_seconds=SMOOTH_WINDOW_SECONDS
    )

    # --------------------------------------------------------
    # Parametri fisici vs temperatura
    # --------------------------------------------------------

    plot_segment_parameter_vs_temperature(
        physical_results,
        time,
        current,
        temperature,
        outlier_threshold=6.0
    )

    # --------------------------------------------------------
    # OCV vs proxy SOC e temperatura
    # --------------------------------------------------------

    plot_ocv_vs_charge_proxy_and_temperature(
        physical_results,
        time,
        current,
        temperature,
        outlier_threshold=6.0
    )

    # --------------------------------------------------------
    # Poli ARX
    # --------------------------------------------------------

    plot_arx_poles_history(
        time,
        rls_results
    )


    # --------------------------------------------------------
    # Validazione quantitativa Digital Twin
    # --------------------------------------------------------

    print_dt_validation_metrics(
        time,
        voltage,
        current,
        temperature,
        rls_results
    )

    plot_dt_validation(
        time,
        voltage,
        current,
        temperature,
        rls_results,
        rms_window_seconds=VALIDATION_RMS_WINDOW_SECONDS
    )

    plot_dt_validation_zoom_windows(
        time,
        voltage,
        current,
        temperature,
        rls_results,
        window_centers=[
            250000,
            500000,
            750000
        ],
        window_seconds=12000
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        rls_results,
        physical_results
    )

    plt.show()