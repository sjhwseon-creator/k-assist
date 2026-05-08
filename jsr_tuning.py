from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

MECH_PATH = Path(r"C:\Users\Hwan\Desktop\RWTH\Projekt\Combustion and Flame\Optimazation Modell\HGD_Seong_V55.yaml")
OUTPUT_DIR = Path(r"C:\Users\Hwan\Desktop\RWTH\Projekt\Simulation\JSR")

# 1-based reaction number range (inclusive)
TUNE_RXN_START_1BASED = 3937
TUNE_RXN_END_1BASED = 4371

A_SCALE_MIN = 0.5
A_SCALE_MAX = 2.0

RESIDENCE_TIME = 1.5
REACTOR_VOLUME = 9.2 * (1e-2) ** 3
PRESSURE_VALVE_COEFF = 0.01
MAX_SIM_TIME = 50.0
INLET_X = {'COC*OOC': 0.01, 'O2': 0.012, 'N2': 0.978}
PRESSURE = ct.one_atm
ENERGY = 'off'

TEMPS = [825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100]
TARGET_SPECIES = ["C2H6", "C2H4", "CH3OH"]

N_ITER = 200
MIN_MUTATIONS_PER_ITER = 1
MAX_MUTATIONS_PER_ITER = 3
LOG_EPS = 1e-12
RANDOM_SEED = 42

# Optional: tune one specific reaction Arrhenius parameters directly (A, Ea)
ENABLE_SPECIAL_EA_A_TUNING = True
SPECIAL_RXN_NUMBER_1BASED = 3937
SPECIAL_EA_MIN = 4.83e4
SPECIAL_EA_MAX = 9.00e4
SPECIAL_A_MIN = 1.0e10
SPECIAL_A_MAX = 1.0e16


@dataclass
class Candidate:
    active_rxn_indices: np.ndarray  # 0-based reaction indices in full mechanism
    active_scales: np.ndarray
    special_A: float | None
    special_Ea: float | None
    loss: float


def experimental_dataframe() -> pd.DataFrame:
    data = {
        "Temperature": TEMPS,
        "C2H6": [0, 0, 0, 4.32e-06, 1.14e-04, 1.68e-04, 1.94e-04, 1.94e-04, 1.88e-04, 1.53e-04, 1.18e-04, 7.95e-05],
        "C2H4": [0, 0, 0, 0, 3.33e-05, 1.12e-04, 1.92e-04, 2.65e-04, 3.68e-04, 4.13e-04, 4.56e-04, 4.45e-04],
        "CH3OH": [4.56e-05, 5.50e-05, 8.78e-05, 1.18e-04, 2.37e-04, 2.50e-04, 2.23e-04, 1.88e-04, 1.65e-04, 9.88e-05, 6.61e-05, 2.49e-05],
    }
    return pd.DataFrame(data).set_index("Temperature")


def make_gas_with_sparse_scales(
    active_rxn_indices: np.ndarray,
    active_scales: np.ndarray,
    special_A: float | None,
    special_Ea: float | None,
) -> ct.Solution:
    gas = ct.Solution(str(MECH_PATH))
    for i in range(gas.n_reactions):
        gas.set_multiplier(1.0, i)
    for rxn_idx, scale in zip(active_rxn_indices, active_scales):
        gas.set_multiplier(float(scale), int(rxn_idx))

    if ENABLE_SPECIAL_EA_A_TUNING:
        if special_A is None or special_Ea is None:
            raise ValueError("special_A and special_Ea are required when ENABLE_SPECIAL_EA_A_TUNING=True")
        special_idx = SPECIAL_RXN_NUMBER_1BASED - 1
        rxn = gas.reaction(special_idx)
        rate = rxn.rate
        if not hasattr(rate, 'temperature_exponent'):
            raise TypeError("Special reaction rate must be Arrhenius-like with A, b, Ea.")
        b = rate.temperature_exponent
        rxn.rate = ct.ArrheniusRate(special_A, b, special_Ea)
        gas.modify_reaction(special_idx, rxn)

    return gas


def run_jsr_for_temperatures(gas: ct.Solution) -> pd.DataFrame:
    concentrations = copy.copy(INLET_X)
    out = pd.DataFrame(index=TEMPS, columns=TARGET_SPECIES, dtype=float)
    for temperature in TEMPS:
        gas.TPX = temperature, PRESSURE, INLET_X
        fuel_tank = ct.Reservoir(gas)
        exhaust = ct.Reservoir(gas)

        gas.TPX = temperature, PRESSURE, concentrations
        reactor = ct.IdealGasReactor(gas, energy=ENERGY, volume=REACTOR_VOLUME)
        ct.MassFlowController(upstream=fuel_tank, downstream=reactor, mdot=reactor.mass / RESIDENCE_TIME)
        ct.Valve(upstream=reactor, downstream=exhaust, K=PRESSURE_VALVE_COEFF)

        net = ct.ReactorNet([reactor])
        t = 0.0
        while t < MAX_SIM_TIME:
            t = net.step()

        concentrations = reactor.thermo.X
        for sp in TARGET_SPECIES:
            out.loc[temperature, sp] = reactor.thermo[sp].X[0]
    return out


def loss_function(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
    log_sim = np.log10(sim_df[TARGET_SPECIES].to_numpy() + LOG_EPS)
    log_exp = np.log10(exp_df[TARGET_SPECIES].to_numpy() + LOG_EPS)
    return float(np.mean((log_sim - log_exp) ** 2))


def random_sparse_candidate(tune_indices: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    if MIN_MUTATIONS_PER_ITER < 1 or MAX_MUTATIONS_PER_ITER < MIN_MUTATIONS_PER_ITER:
        raise ValueError("Invalid mutation count bounds")
    k = rng.randint(MIN_MUTATIONS_PER_ITER, MAX_MUTATIONS_PER_ITER)
    k = min(k, len(tune_indices))
    chosen = np.array(rng.sample(list(tune_indices), k=k), dtype=int)
    scales = np.array([rng.uniform(A_SCALE_MIN, A_SCALE_MAX) for _ in range(k)], dtype=float)

    special_A = None
    special_Ea = None
    if ENABLE_SPECIAL_EA_A_TUNING:
        special_A = 10 ** rng.uniform(np.log10(SPECIAL_A_MIN), np.log10(SPECIAL_A_MAX))
        special_Ea = rng.uniform(SPECIAL_EA_MIN, SPECIAL_EA_MAX)
    return chosen, scales, special_A, special_Ea


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_gas = ct.Solution(str(MECH_PATH))
    if TUNE_RXN_START_1BASED < 1:
        raise ValueError("TUNE_RXN_START_1BASED must be >= 1.")
    if TUNE_RXN_END_1BASED < TUNE_RXN_START_1BASED:
        raise ValueError("TUNE_RXN_END_1BASED must be >= TUNE_RXN_START_1BASED.")
    if TUNE_RXN_END_1BASED > base_gas.n_reactions:
        raise ValueError(f"Requested range end {TUNE_RXN_END_1BASED} exceeds mechanism reactions {base_gas.n_reactions}.")

    tune_indices = np.arange(TUNE_RXN_START_1BASED - 1, TUNE_RXN_END_1BASED, dtype=int)
    if ENABLE_SPECIAL_EA_A_TUNING and not (1 <= SPECIAL_RXN_NUMBER_1BASED <= base_gas.n_reactions):
        raise ValueError("SPECIAL_RXN_NUMBER_1BASED is out of mechanism reaction range.")
    exp_df = experimental_dataframe()

    best = Candidate(active_rxn_indices=np.array([], dtype=int), active_scales=np.array([], dtype=float), special_A=None, special_Ea=None, loss=float("inf"))
    best_sim = None
    history = []
    trial_history = []

    rng = random.Random(RANDOM_SEED)
    t0 = time.time()

    for it in range(1, N_ITER + 1):
        active_rxn_indices, active_scales, special_A, special_Ea = random_sparse_candidate(tune_indices, rng)
        trial_gas = make_gas_with_sparse_scales(active_rxn_indices, active_scales, special_A, special_Ea)
        trial_sim = run_jsr_for_temperatures(trial_gas)
        trial_loss = loss_function(trial_sim, exp_df)

        trial_history.append({
            "iteration": it,
            "trial_loss": trial_loss,
            "num_tuned_reactions": len(active_rxn_indices),
            "tuned_reaction_numbers_1based": ";".join(str(int(x + 1)) for x in active_rxn_indices),
            "tuned_A_multipliers": ";".join(f"{x:.12g}" for x in active_scales),
            "special_reaction_number_1based": SPECIAL_RXN_NUMBER_1BASED if ENABLE_SPECIAL_EA_A_TUNING else "",
            "special_A": special_A if ENABLE_SPECIAL_EA_A_TUNING else "",
            "special_Ea": special_Ea if ENABLE_SPECIAL_EA_A_TUNING else "",
        })

        if trial_loss < best.loss:
            best = Candidate(active_rxn_indices=active_rxn_indices, active_scales=active_scales, special_A=special_A, special_Ea=special_Ea, loss=trial_loss)
            best_sim = trial_sim
            print(f"Iter {it:04d}: improved loss={best.loss:.6e}, tuned_rxns={len(best.active_rxn_indices)}")

        history.append((it, best.loss, len(best.active_rxn_indices)))

    elapsed = time.time() - t0
    print(f"Optimization done in {elapsed:.1f} s")

    if best_sim is None:
        raise RuntimeError("No candidate was evaluated.")

    best_sim.to_csv(OUTPUT_DIR / "best_species_fit.csv", index=True)
    exp_df.to_csv(OUTPUT_DIR / "exp_species_target.csv", index=True)
    pd.DataFrame(history, columns=["iteration", "best_loss", "best_num_tuned_reactions"]).to_csv(
        OUTPUT_DIR / "loss_history.csv", index=False
    )
    pd.DataFrame(trial_history).to_csv(OUTPUT_DIR / "trial_parameter_history.csv", index=False)

    scale_df = pd.DataFrame({
        "reaction_number_1based": best.active_rxn_indices + 1,
        "A_multiplier": best.active_scales,
    }).sort_values("reaction_number_1based")
    multiplier_file = OUTPUT_DIR / f"best_A_multipliers_sparse_{TUNE_RXN_START_1BASED}_{TUNE_RXN_END_1BASED}.csv"
    scale_df.to_csv(multiplier_file, index=False)

    if ENABLE_SPECIAL_EA_A_TUNING:
        pd.DataFrame([{"reaction_number_1based": SPECIAL_RXN_NUMBER_1BASED, "A": best.special_A, "Ea": best.special_Ea}]).to_csv(
            OUTPUT_DIR / "best_special_reaction_A_Ea.csv", index=False
        )

    merged = best_sim.copy()
    for sp in TARGET_SPECIES:
        merged[f"{sp}_exp"] = exp_df[sp]
        merged[f"{sp}_sim"] = best_sim[sp]
    merged.to_csv(OUTPUT_DIR / "fit_comparison.csv", index=True)

    print("Saved files:")
    print(multiplier_file)
    print(OUTPUT_DIR / "trial_parameter_history.csv")


if __name__ == "__main__":
    main()
