import numpy as np
import openmc
from neutronics_calphad.neutronics.dose import contact_dose
from neutronics_calphad.neutronics.depletion import extract_gas_production

# default “critical” gas limits (in appm)
CRITICAL_GAS_LIMITS = {
    "He_appm": 1000,
    "H_appm": 500
}

# mapping cooling times (in days after start of cooling) → maximum allowed Sv/h/kg
DOSE_LIMITS = {
    14:     1e2,    # 14 days → 100 Sv/h/kg
    365:    1e-2,  # 1 year → 0.01 Sv/h/kg
    3650:   1e-2,  # 10 years → 0.01 Sv/h/kg
    36500:  1e-4   # 100 years → 0.0001 Sv/h/kg
}


def evaluate_material(
    results: openmc.deplete.Results,
    chain_file: str,
    abs_file: str,
    critical_gas_limits: dict = CRITICAL_GAS_LIMITS,
    dose_limits: dict = DOSE_LIMITS
) -> float:
    """
    Evaluate a depletion result for both contact dose and gas‐production limits.

    Returns:
        1.0 if all limits passed,
        0.5 if either dose OR gas passed,
        0.0 if both failed.
    """

    # 1) Compute full dose‐time series
    times_s, dose_dicts = contact_dose(results=results, chain_file=chain_file, abs_file=abs_file)

    # 2) Compute gas production (appm)
    gas_production_rates = extract_gas_production(results)

    # 3) Identify end of irradiation (last non-zero source rate)
    source_rates = results.get_source_rates()
    final_idx = np.nonzero(source_rates)[0][-1]
    final_irr_time = times_s[final_idx]

    print(f"Source rates: {source_rates}")
    print(f"Final source‐rate index: {final_idx}")
    print(f"End of irradiation at t = {final_irr_time:.0f}s ({final_irr_time/3600/24:.1f} days)")

    # 4) Build a total‐dose lookup
    total_dose = {t: sum(d.values()) for t, d in zip(times_s, dose_dicts)}

    # 5) Locate the first cooling step
    cool_start_idx = final_idx + 1
    if cool_start_idx >= len(times_s):
        raise RuntimeError("No post‐irradiation time steps available for cooling‐time checks.")

    cool_start_time = times_s[cool_start_idx]
    print(f"Cooling starts at t = {cool_start_time:.0f}s "
          f"({(cool_start_time - final_irr_time)/3600/24:.1f} days after irrad.)")

    # 6) Build cooling‐time array and corresponding doses
    cool_times  = times_s[cool_start_idx:] - cool_start_time
    cool_doses  = [total_dose[t] for t in times_s[cool_start_idx:]]

    # 7) Check each cooling‐time limit
    satisfy_dose = True
    for days_after, limit in dose_limits.items():
        target_s = days_after * 24 * 3600
        # first index in cool_times >= target_s
        rel_idx = np.searchsorted(cool_times, target_s, side='left')
        if rel_idx >= len(cool_times):
            rel_idx = len(cool_times) - 1

        abs_idx    = cool_start_idx + rel_idx
        t_actual   = times_s[abs_idx]
        cool_actual= cool_times[rel_idx]
        rate_actual= total_dose[t_actual]

        print(
            f"Dose @ {t_actual:.0f}s "
            f"({cool_actual/3600/24:.1f} days after cooling start) = "
            f"{rate_actual:.2e} Sv/h/kg   vs limit {limit:.2e} Sv/h/kg"
        )

        if rate_actual > limit:
            satisfy_dose = False

    # 8) Check gas‐production limits
    satisfy_gas = True
    for gas, produced in gas_production_rates.items():
        limit = critical_gas_limits.get(gas, np.inf)
        print(f"Gas production: {produced:.2f} appm {gas} (limit {limit} appm)")
        if produced > limit:
            satisfy_gas = False

    # 9) Return fitness
    if satisfy_dose and satisfy_gas:
        return 1.0
    elif satisfy_dose or satisfy_gas:
        return 0.5
    else:
        return 0.0
