import subprocess

# monkeytype.sqlite3 database with potential type hits is generated
# RUN IN TERMINAL: "monkeytype run runtests.py"

# lists all available modules
# that might be type hinted
# RUN IN TERMINAL:  "monkeytype list-modules"

# result of list-modules
targets = [
    # "utils.test_legacy_support",
    # "utils.test_iteration",
    # "utils.test_ffts",
    # "utils.test_data_check",
    # "utils.test_blondmath",
    # "trackers.test_tracker",
    # "trackers.test_drift",
    # "test_installation",
    # "synchrotron_radiation.test_synch_rad",
    # "llrf.test_signal_processing",
    # "llrf.test_rf_modulation",
    # "llrf.test_offset_frequency",
    # "llrf.test_impulse_response",
    # "llrf.test_cavity_feedback",
    # "interfaces.rf_noise_cpp.test_wrap_rf_noise",
    # "integration.test_validate_mpi",
    # "integration.test_validate_gpu",
    # "integration.test_mpi_examples",
    # "integration.test_gpu_examples",
    # "integration.test_examples",
    # "input_parameters.test_ring",
    # "input_parameters.test_rf_params_object",
    # "input_parameters.test_preprocess",
    # "impedances.test_impedance_sources",
    # "impedances.test_impedance",
    # "general.test_separatrix_bigaussian",
    "blond.utils.track_iteration",
    "blond.utils.legacy_support",
    "blond.utils.data_check",
    "blond.utils.butils_wrap_cpp",
    "blond.utils.bmath",
    "blond.utils",
    "blond.trackers.utilities",
    "blond.trackers.tracker",
    "blond.toolbox.next_regular",
    "blond.toolbox.filters_and_fitting",
    "blond.synchrotron_radiation.synchrotron_radiation",
    "blond.llrf.signal_processing",
    "blond.llrf.rf_modulation",
    "blond.llrf.offset_frequency",
    "blond.llrf.impulse_response",
    "blond.llrf.cavity_feedback",
    "blond.llrf.beam_feedback",
    "blond.interfaces.rf_noise_cpp.wrap_rf_noise",
    "blond.input_parameters.ring_options",
    "blond.input_parameters.ring",
    "blond.input_parameters.rf_parameters_options",
    "blond.input_parameters.rf_parameters",
    "blond.impedances.impedance_sources",
    "blond.impedances.impedance",
    "blond.gpu",
    "blond.beam.sparse_slices",
    "blond.beam.profile",
    "blond.beam.distributions",
    "blond.beam.coasting_beam",
    "blond.beam.beam",
    # "beams.test_coasting_beam",
    # "beams.test_beam_object",
    # "beam_profile.test_sparse_profile",
    # "beam_profile.test_beam_profile_object",
    # "beam.test_beam",

]
# add type hints from database to files
for target in targets:
    subprocess.run(f'monkeytype apply {target}'.split(" "))
