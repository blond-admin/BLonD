import inspect
from typing import Callable
from warnings import warn


def _handle_legacy_kwargs(new_by_old: dict[str, str]):
    """Handles renamed keyword arguments and warns user

    Parameters
    ----------
    new_by_old: dict
        dictionary like {old1 : new1, ...}

    """

    def decorator(func: Callable):
        def wrapped(*args, **kwargs_potentially_old):
            kwargs_fixed = {}
            for (
                key_potentially_old,
                argument,
            ) in kwargs_potentially_old.items():
                if key_potentially_old in new_by_old:
                    warn(
                        f"Keyword argument '{key_potentially_old}' when calling "
                        f"'{func.__name__}' is deprecated. Use '{new_by_old[key_potentially_old]}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                kwargs_fixed[
                    new_by_old.get(key_potentially_old, key_potentially_old)
                ] = argument
            return func(*args, **kwargs_fixed)

        return wrapped

    return decorator


# general replacements that were performed in blond
__new_by_old = {
    "Beam": "beam",
    "FitOptions": "fit_options",
    "FilterOptions": "filter_options",
    "OtherSlicesOptions": "other_slices_options",
    "Ring": "ring",
    "GeneralParameters": "ring",
    "rf_parameters": "rf_station",
    "RFStation": "rf_station",
    "RFParams": "rf_station",
    "RFSectionParameters": "rf_station",
    "ghostBeam": "ghost_beam",
    "RFStationOptions": "rf_station_options",
    "Particle": "particle",
    "System": "system",
    "CutOptions": "cut_options",
    "Profile": "profile",
    "PhaseLoop": "phase_loop",
    "PL": "phase_loop",
    "rf": "rf_station",
    "Nbunches": "n_bunches",
    "TotalInducedVoltage": "total_induced_voltage",
    "RingOptions": "ring_options",
    "full_ring_and_RF": "full_ring_and_rf",
    "extraVoltageDict": "extra_voltage_dict",
    "FullRingAndRF": "full_ring_and_rf",
    "filterMethod": "filter_method",
    "filterExtraOptions": "filter_extra_options",
    "MainH": "main_harmonic",
    "main_h": "main_harmonic",
    "NewFrequencyProgram": "new_frequency_program",
    "FixedFrequency": "fixed_frequency",
    "FixedDuration": "fixed_duration",
    "TransitionDuration": "transition_duration",
    "omegaProg": "omega_prog",
    "LHCNoiseFB": "lhc_noise_feedback",
    "Y_array": "y_array",
    "X_array": "x_array",
    "fitExtraOptions": "fit_extra_options",
    "Resonators": "resonators",
    "RingAndRFSection_list": "ring_and_rf_section",
    "BeamFeedback": "beam_feedback",
    "CavityFeedback": "cavity_feedback",
    "smoothOption": "smooth_option",
    "timeArray": "time_array",
}
handle_legacy_kwargs = _handle_legacy_kwargs(__new_by_old)
