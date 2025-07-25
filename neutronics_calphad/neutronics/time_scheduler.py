import numpy as np
from typing import List, Tuple, Optional

class TimeScheduler:
    """
    Simple, modular scheduler for OpenMC depletion timesteps.
    """

    UNITS_IN_SECONDS = {
        'sec': 1,
        'second': 1,
        'seconds': 1,
        'min': 60,
        'minute': 60,
        'minutes': 60,
        'hour': 3600,
        'hours': 3600,
        'hr': 3600,
        'day': 24 * 3600,
        'days': 24 * 3600,
        'week': 7 * 24 * 3600,
        'weeks': 7 * 24 * 3600,
        'month': 30 * 24 * 3600,      # Approximate
        'months': 30 * 24 * 3600,
        'year': 365.25 * 24 * 3600,   # Includes leap days
        'years': 365.25 * 24 * 3600,
    }

    def __init__(
        self,
        irradiation_time: str,
        cooling_times: List[str],
        source_rate: Optional[float] = None,
        irradiation_steps: int = 1
    ) -> None:
        """
        Build a timestep schedule.

        Args:
            irradiation_time: Time spec for irradiation (e.g. "1 year")
            cooling_times: List of cooling specs (e.g. ["1 day", "1 week", ...])
            source_rate: Neutron source rate during irradiation (neutrons/sec). If None, no source rate is tracked.
            irradiation_steps: Number of steps to split the irradiation period (default 1).
        """
        irrad_s = self.parse_time_with_units(irradiation_time)
        cooling_s = sorted({self.parse_time_with_units(t) for t in cooling_times})

        self.irradiation_time: float = irrad_s
        self.cooling_times: List[float] = cooling_s
        self.source_rate: Optional[float] = source_rate
        self.irradiation_steps: int = irradiation_steps

        # first timestep is irradiation (possibly split)
        durations: List[float] = []
        if irradiation_steps < 1:
            raise ValueError("irradiation_steps must be >= 1")
        irrad_step = irrad_s / irradiation_steps
        durations.extend([irrad_step] * irradiation_steps)

        prev = 0.0
        for ct in cooling_s:
            if ct <= prev:
                raise ValueError(f"Cooling times must increase: {ct} <= {prev}")
            durations.append(ct - prev)
            prev = ct

        self.timestep_durations: np.ndarray = np.array(durations)
        self.cumulative_times: np.ndarray = np.cumsum(self.timestep_durations)

    @classmethod
    def create_standard_cooling_schedule(
        cls,
        source_rate: Optional[float] = None,
        irradiation_steps: int = 1
    ) -> "TimeScheduler":
        """
        Standard fusion-material schedule:
        1 year irradiation, then cooling at:
            1 sec, 1 hr, 10 hr, 1 day, 1 wk, 2 wk,
            1 mo, 2 mo, 1 yr, 5 yr, 10 yr, 25 yr, 100 yr

        Args:
            source_rate: Neutron source rate during irradiation (neutrons/sec). If None, no source rate is tracked.
            irradiation_steps: Number of steps to split the irradiation period (default 1).
        Returns:
            TimeScheduler instance with standard schedule.
        """
        cooling = [
            "1 sec", "1 hour", "10 hours", "1 day", "1 week",
            "2 weeks", "1 month", "2 months",
            "1 year", "5 years", "10 years",
            "25 years", "100 years"
        ]
        return cls("1 year", cooling, source_rate=source_rate, irradiation_steps=irradiation_steps)

    @staticmethod
    def parse_time_with_units(time_spec: str) -> float:
        """
        Parse a string like "2 weeks" into seconds.

        Args:
            time_spec: String with number and unit (e.g. "2 weeks").
        Returns:
            Time in seconds.
        """
        parts = time_spec.strip().split()
        if len(parts) != 2:
            raise ValueError(f"Expected '<number> <unit>', got '{time_spec}'")
        value_str, unit_str = parts
        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"Invalid number: '{value_str}'")
        unit = unit_str.lower()
        if unit not in TimeScheduler.UNITS_IN_SECONDS:
            raise ValueError(f"Unknown unit '{unit}'. Supported: {list(TimeScheduler.UNITS_IN_SECONDS)}")
        return value * TimeScheduler.UNITS_IN_SECONDS[unit]

    @staticmethod
    def format_time_duration(seconds: float) -> str:
        """
        Format seconds into human-readable string.

        Args:
            seconds: Time duration in seconds.
        Returns:
            Human-readable string.
        """
        if seconds < 60:
            return f"{seconds:.1f} sec"
        elif seconds < 3600:
            return f"{seconds/60:.1f} min"
        elif seconds < 24*3600:
            return f"{seconds/3600:.1f} hours"
        elif seconds < 365.25*24*3600:
            return f"{seconds/(24*3600):.1f} days"
        else:
            return f"{seconds/(365.25*24*3600):.1f} years"

    def print_schedule(self) -> None:
        """
        Print the schedule in a table.
        """
        header = f"{'Step':<4} {'Duration':<15} {'Cumulative':<15} {'Phase':<12}"
        sep = "-" * len(header)
        print("Time Schedule")
        print(sep)
        print(header)
        print(sep)

        for i, (dur, cum) in enumerate(zip(self.timestep_durations, self.cumulative_times), start=1):
            phase = "Irradiation" if i <= self.irradiation_steps else "Cooling"
            dur_str = self.format_time_duration(dur)
            cum_str = self.format_time_duration(cum)
            print(f"{i:<4} {dur_str:<15} {cum_str:<15} {phase:<12}")

        print(sep)

    def get_timesteps_and_source_rates(self) -> Tuple[List[float], List[float]]:
        """
        Get the list of non-cumulative timesteps and corresponding source rates.

        Returns:
            Tuple of (timesteps, source_rates), both lists of floats in seconds and neutrons/sec.
            The length of both lists is equal to the number of steps (irradiation + cooling).
            Irradiation steps have the specified source_rate, cooling steps have 0.
        """
        timesteps = self.timestep_durations.tolist()
        if self.source_rate is not None:
            source_rates = [self.source_rate] * self.irradiation_steps + [0.0] * (len(timesteps) - self.irradiation_steps)
        else:
            source_rates = [0.0] * len(timesteps)
        return timesteps, source_rates
