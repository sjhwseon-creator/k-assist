from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd

MECH_PATH = Path(r"C:\Users\Hwan\Desktop\RWTH\Projekt\Combustion and Flame\Optimazation Modell\HGD_Seong_V55.yaml")
OUTPUT_DIR = Path(r"C:\Users\Hwan\Desktop\RWTH\Projekt\Simulation\JSR")

TARGET_RXN_NUMBER_1BASED = 3937
A_MIN = 1.0e10
A_MAX = 1.0e16
EA_MIN = 4.83e4
EA_MAX = 9.00e4

RESIDENCE_TIME = 1.5
REACTOR_VOLUME = 9.2 * (1e-2) ** 3
PRESSURE_VALVE_COEFF = 0.01
MAX_SIM_TIME = 50.0
INLET_X = {'COC*OOC': 0.01, 'O2': 0.012, 'N2': 0.978}
PRESSURE = ct.one_atm
ENERGY = 'off'

TEMPS = [825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100]
TARGET_SPECIES = ["C2H6", "C2H4", "CH3OH"]

# Constraint setting:
# 1) anchor species must be within +/-10%
# 2) other species must remain within factor window [0.5, 2.0]
ANCHOR_SPECIES_LIST = ["CH3OH"]
ANCHOR_REL_TOL = 0.10
OTHER_FACTOR_MIN = 0.5
OTHER_FACTOR_MAX = 2.0
CONSTRAINT_PENALTY = 1e3

N_ITER = 50
LOG_EPS = 1e-12
RANDOM_SEED = 42


def experimental_dataframe() -> pd.DataFrame:
    data = {
        "Temperature": TEMPS,
        "C2H6": [0, 0, 0, 4.32e-06, 1.14e-04, 1.68e-04, 1.94e-04, 1.94e-04, 1.88e-04, 1.53e-04, 1.18e-04, 7.95e-05],
        "C2H4": [0, 0, 0, 0, 3.33e-05, 1.12e-04, 1.92e-04, 2.65e-04, 3.68e-04, 4.13e-04, 4.56e-04, 4.45e-04],
        "CH3OH": [4.56e-05, 5.50e-05, 8.78e-05, 1.18e-04, 2.37e-04, 2.50e-04, 2.23e-04, 1.88e-04, 1.65e-04, 9.88e-05, 6.61e-05, 2.49e-05],
    }
    return pd.DataFrame(data).set_index("Temperature")


def apply_single_reaction_inplace(gas: ct.Solution, rxn_index_zero_based: int, b_value: float, A_value: float, Ea_value: float) -> None:
    rxn = gas.reaction(rxn_index_zero_based)
    rxn.rate = ct.ArrheniusRate(A_value, b_value, Ea_value)
    gas.modify_reaction(rxn_index_zero_based, rxn)


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


def constrained_loss(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> tuple[float, bool, float, float]:
    # Base objective in log-space
    log_sim = np.log10(sim_df[TARGET_SPECIES].to_numpy() + LOG_EPS)
    log_exp = np.log10(exp_df[TARGET_SPECIES].to_numpy() + LOG_EPS)
    base_loss = float(np.mean((log_sim - log_exp) ** 2))

    # 1) Anchor species must be within +/-10%
    anchor_violation_terms = []
    for anchor_sp in ANCHOR_SPECIES_LIST:
        sim_anchor = sim_df[anchor_sp].to_numpy()
        exp_anchor = exp_df[anchor_sp].to_numpy()
        anchor_rel_err = np.abs(sim_anchor - exp_anchor) / np.maximum(np.abs(exp_anchor), LOG_EPS)
        anchor_violation_terms.append(np.maximum(anchor_rel_err - ANCHOR_REL_TOL, 0.0))
    anchor_violation = np.vstack(anchor_violation_terms)

    # 2) Other species must stay in [0.5, 2.0] factor window
    others = [sp for sp in TARGET_SPECIES if sp not in ANCHOR_SPECIES_LIST]
    ratio_violation_sum = 0.0
    for sp in others:
        sim = sim_df[sp].to_numpy()
        exp = exp_df[sp].to_numpy()
        ratio = (sim + LOG_EPS) / (exp + LOG_EPS)
        lower_v = np.maximum(OTHER_FACTOR_MIN - ratio, 0.0)
        upper_v = np.maximum(ratio - OTHER_FACTOR_MAX, 0.0)
        ratio_violation_sum += float(np.mean(lower_v + upper_v))

    anchor_violation_mean = float(np.mean(anchor_violation))
    total_violation = anchor_violation_mean + ratio_violation_sum
    feasible = total_violation <= 0.0

    total_loss = base_loss + CONSTRAINT_PENALTY * total_violation
    return total_loss, feasible, base_loss, total_violation


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gas = ct.Solution(str(MECH_PATH))
    if not (1 <= TARGET_RXN_NUMBER_1BASED <= gas.n_reactions):
        raise ValueError(f"TARGET_RXN_NUMBER_1BASED must be in [1, {gas.n_reactions}].")
    invalid_anchors = [sp for sp in ANCHOR_SPECIES_LIST if sp not in TARGET_SPECIES]
    if invalid_anchors:
        raise ValueError(f"All anchors must be in TARGET_SPECIES. Invalid: {invalid_anchors}")

    rxn_index = TARGET_RXN_NUMBER_1BASED - 1
    rxn = gas.reaction(rxn_index)
    rate = rxn.rate
    if not hasattr(rate, "temperature_exponent"):
        raise TypeError("Target reaction must be Arrhenius-like with A, b, Ea.")
    b_value = rate.temperature_exponent

    exp_df = experimental_dataframe()

    best_total_loss = float("inf")
    best_A = None
    best_Ea = None
    best_sim = None
    best_base_loss = None
    best_violation = None
    trial_history = []

    t0 = time.time()
    for it in range(1, N_ITER + 1):
        A_value = 10 ** random.uniform(np.log10(A_MIN), np.log10(A_MAX))
        Ea_value = random.uniform(EA_MIN, EA_MAX)

        apply_single_reaction_inplace(gas, rxn_index, b_value, A_value, Ea_value)
        sim_df = run_jsr_for_temperatures(gas)
        total_loss, feasible, base_loss, violation = constrained_loss(sim_df, exp_df)

        trial_history.append({
            "iteration": it,
            "A": A_value,
            "Ea": Ea_value,
            "total_loss": total_loss,
            "base_loss": base_loss,
            "constraint_violation": violation,
            "feasible": feasible,
        })

        improved = total_loss < best_total_loss
        if improved:
            best_total_loss = total_loss
            best_A = A_value
            best_Ea = Ea_value
            best_sim = sim_df
            best_base_loss = base_loss
            best_violation = violation

        status = "*best*" if improved else ""
        print(
            f"Iter {it:04d}: total_loss={total_loss:.6e}, base_loss={base_loss:.6e}, "
            f"viol={violation:.3e}, feasible={feasible}, A={A_value:.6e}, Ea={Ea_value:.6e} {status}".rstrip()
        )

    elapsed = time.time() - t0
    print(f"Optimization done in {elapsed:.1f} s")

    if best_sim is None:
        raise RuntimeError("No trial evaluated.")

    best_sim.to_csv(OUTPUT_DIR / "single_rxn_best_species_fit.csv", index=True)
    exp_df.to_csv(OUTPUT_DIR / "single_rxn_exp_species_target.csv", index=True)
    pd.DataFrame(trial_history).to_csv(OUTPUT_DIR / "single_rxn_trial_history.csv", index=False)
    pd.DataFrame([{
        "reaction_number_1based": TARGET_RXN_NUMBER_1BASED,
        "best_A": best_A,
        "best_Ea": best_Ea,
        "best_total_loss": best_total_loss,
        "best_base_loss": best_base_loss,
        "best_constraint_violation": best_violation,
        "anchor_species_list": ";".join(ANCHOR_SPECIES_LIST),
        "anchor_rel_tol": ANCHOR_REL_TOL,
        "other_factor_min": OTHER_FACTOR_MIN,
        "other_factor_max": OTHER_FACTOR_MAX,
    }]).to_csv(OUTPUT_DIR / "single_rxn_best_A_Ea.csv", index=False)

    merged = best_sim.copy()
    for sp in TARGET_SPECIES:
        merged[f"{sp}_exp"] = exp_df[sp]
        merged[f"{sp}_sim"] = best_sim[sp]
    merged.to_csv(OUTPUT_DIR / "single_rxn_fit_comparison.csv", index=True)


if __name__ == "__main__":
    main()
