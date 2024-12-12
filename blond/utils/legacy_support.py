from typing import Callable, Dict
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
            for key_potentially_old, argument in kwargs_potentially_old.items():
                if key_potentially_old in new_by_old:
                    warn(
                        f"Keyword argument '{key_potentially_old}' when calling "
                        f"'{func.__name__}' is deprecated. Use '{new_by_old[key_potentially_old]}' instead.",
                        DeprecationWarning
                    )
                key_potentially_new = new_by_old.get(key_potentially_old, key_potentially_old)
                if isinstance(key_potentially_new, Callable):
                    kwargs_fixed.update(key_potentially_new(argument))
                else:
                    kwargs_fixed[key_potentially_new] = argument
            return func(*args, **kwargs_fixed)

        return wrapped

    return decorator


def convert_distribution_options_list(distribution_options_list: list[dict] | dict):
    if isinstance(distribution_options_list, list):
        tmp = [kwargs_by_distribution_options(item) for item in distribution_options_list]
        fits = [x["fit"] for x in tmp]
        distribution_variables = [x["distribution_variable"] for x in tmp]

        # assert that all distribution variables are the same
        for dv in distribution_variables:
            assert dv == distribution_variables[0], "distribution_variable cant change!"

        return {"fit": fits, "distribution_variable": distribution_variables[0]}
    elif isinstance(distribution_options_list, dict):
        return kwargs_by_distribution_options(distribution_options_list)
    else:
        raise RuntimeError("Something went wrong ! ")


def kwargs_by_distribution_options(distribution_options: dict):
    if 'type' in distribution_options:
        distribution_type = distribution_options['type']
    else:
        distribution_type = None

    if 'exponent' in distribution_options:
        distribution_exponent = distribution_options['exponent']
    else:
        distribution_exponent = None

    if 'emittance' in distribution_options:
        emittance = distribution_options['emittance']
    else:
        emittance = None

    if 'bunch_length' in distribution_options:
        bunch_length = distribution_options['bunch_length']
    else:
        bunch_length = None

    if 'bunch_length_fit' in distribution_options:
        bunch_length_fit = distribution_options['bunch_length_fit']
    else:
        bunch_length_fit = None

    if 'density_variable' in distribution_options:
        distribution_variable = distribution_options['density_variable']
    else:
        distribution_variable = None

    if distribution_options['type'] == 'user_input':
        distribution_function_input = distribution_options['function']
    else:
        distribution_function_input = None

    if distribution_options['type'] == 'user_input_table':
        distribution_user_table = {
            'user_table_action': distribution_options['user_table_action'],
            'user_table_density': distribution_options['user_table_density']}
    else:
        distribution_user_table = None
    if distribution_user_table is not None:
        from blond.beam.distribution_generators.singlebunch.matched_from_distribution_function import \
            FitDistributionUserTable
        fit = FitDistributionUserTable(
            distribution_user_table=distribution_user_table
        )
    elif emittance is not None:
        from blond.beam.distribution_generators.singlebunch.matched_from_distribution_function import FitEmittance
        fit = FitEmittance(
            emittance=emittance,
            distribution_type=distribution_type,
            distribution_exponent=distribution_exponent,
        )
        fit.distribution_function_input = distribution_function_input
    elif bunch_length is not None:
        from blond.beam.distribution_generators.singlebunch.matched_from_distribution_function import FitBunchLength
        fit = FitBunchLength(
            bunch_length=bunch_length,
            bunch_length_fit=bunch_length_fit,
            distribution_type=distribution_type,
            distribution_exponent=distribution_exponent,
        )
        fit.distribution_function_input = distribution_function_input
    else:
        raise RuntimeError()
    return {"fit": fit, "distribution_variable": distribution_variable}


def convert_line_density_options_list(line_density_options_list: list[dict] | dict):
    if isinstance(line_density_options_list, list):
        fits = [kwargs_by_line_density_options(item)["fit"]
               for item in line_density_options_list]

        return {"fit": fits}
    elif isinstance(line_density_options_list, dict):
        return kwargs_by_line_density_options(line_density_options_list)
    else:
        raise RuntimeError("Something went wrong !")


def kwargs_by_line_density_options(line_density_options):
    if 'bunch_length' in line_density_options:
        bunch_length = line_density_options['bunch_length']
    else:
        bunch_length = None

    if 'type' in line_density_options:
        line_density_type = line_density_options['type']
    else:
        line_density_type = None

    if 'exponent' in line_density_options:
        line_density_exponent = line_density_options['exponent']
    else:
        line_density_exponent = None

    if line_density_options['type'] == 'user_input':
        line_density_input = {
            'time_line_den': line_density_options['time_line_den'],
            'line_density': line_density_options['line_density']}
    else:
        line_density_input = None

    from blond.beam.distribution_generators.singlebunch.matched_from_line_density import FitLineDensityInput, \
        FitBunchLength
    if line_density_input is not None:
        fit = FitLineDensityInput(line_density_input=line_density_input)
    elif bunch_length is not None:
        fit = FitBunchLength(
            bunch_length=bunch_length,
            line_density_type=line_density_type,
            line_density_exponent=line_density_exponent
        )
    else:
        raise RuntimeError("Something went wrong !")
    return {"fit": fit}


# general replacements that were performed in blond
__new_by_old = {
    "distribution_options_list": convert_distribution_options_list,
    "line_density_options_list": convert_line_density_options_list,
    "Beam": "beam",
    "FitOptions": "fit_options",
    "FilterOptions": "filter_options",
    "OtherSlicesOptions": "other_slices_options",
    "Ring": "ring",
    "GeneralParameters": "ring",
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
    "smoothOption": "smooth_option"
}

handle_legacy_kwargs = _handle_legacy_kwargs(__new_by_old)
