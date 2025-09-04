import subprocess

# RUN IN TERMINAL: "monkeytype run run_pytest.py"
# monkeytype.sqlite3 database with potential type hits is generated

# lists all available modules
# that might be type-hinted
# RUN IN TERMINAL:  "monkeytype list-modules"

# result of list-modules, might need to be updated
targets = """
unittests.test_acc_math.test_analytic.test_hammilton
unittests.physics.test_drifts
unittests.physics.test_cavities
unittests.physics.impedances.test_sovlers
unittests.physics.impedances.test_sources
unittests.physics.impedances.comparisons.test_resonator
unittests.physics.impedances.comparisons.test_inductive
unittests.physics.impedances.compare_with_legacy.test_integration_InductiveImpedance
unittests.physics.impedances.compare_with_legacy.test_integration_InducedVoltageFreq
unittests.physics.feedbacks.test_helpers
unittests.physics.feedbacks.accelerators.sps.test_impulse_response
unittests.physics.feedbacks.accelerators.sps.test_helpers
unittests.physics.feedbacks.accelerators.sps.test_cavity_feedback
unittests.physics.feedbacks.accelerators.sps.test_beam_feedback
unittests.physics.feedbacks.accelerators.lhc.test_cavity_feedback
unittests.handle_results.test_helpers
unittests.handle_results.test_array_recorders
unittests.examples.test_EX_MuonCollider
unittests.examples.test_EX_05_Wake_impedance
unittests.examples.test_EX_04_Stationary_multistation
unittests.examples.test_EX_03_RFnoise
unittests.examples.test_EX_02_Main_long_ps_booster
unittests.examples.test_EX_01_Acceleration
unittests.cycles.test_energy_cycle
unittests.cycles.test_base
unittests.beam_preparation.test_bigaussian
unittests._generals.test_iterables
unittests._core.test_helpers
unittests._core.test_base
unittests._core.simulation.test_simulation
unittests._core.ring.test_ring
unittests._core.ring.test_helpers
unittests._core.ring.test_beam_physics_relevant_elements
unittests._core.beam.test_beams
unittests._core.beam.test_base
unittests._core.backends.test_backend
unittests._core.backends.python.test_callables
blond.testing.simulation
blond.physics.profiles
blond.physics.impedances.sources
blond.physics.impedances.solvers
blond.physics.impedances.readers
blond.physics.impedances.base
blond.physics.feedbacks.helpers
blond.physics.feedbacks.cavity_feedback
blond.physics.feedbacks.beam_feedback
blond.physics.feedbacks.base
blond.physics.feedbacks.accelerators.sps.impulse_response
blond.physics.feedbacks.accelerators.sps.helpers
blond.physics.feedbacks.accelerators.sps.cavity_feedback
blond.physics.feedbacks.accelerators.sps.beam_feedback
blond.physics.feedbacks.accelerators.lhc.helpers
blond.physics.feedbacks.accelerators.lhc.cavity_feedback
blond.physics.drifts
blond.physics.cavities
blond.legacy.blond2.utils.legacy_support
blond.legacy.blond2.utils.butils_wrap_numba
blond.legacy.blond2.utils.butils_wrap_cpp
blond.legacy.blond2.utils.bmath_backends
blond.legacy.blond2.utils
blond.legacy.blond2.trackers.utilities
blond.legacy.blond2.trackers.tracker
blond.legacy.blond2.toolbox.next_regular
blond.legacy.blond2.llrf.impulse_response
blond.legacy.blond2.llrf.beam_feedback
blond.legacy.blond2.input_parameters.ring_options
blond.legacy.blond2.input_parameters.ring
blond.legacy.blond2.input_parameters.rf_parameters_options
blond.legacy.blond2.input_parameters.rf_parameters
blond.legacy.blond2.impedances.induced_voltage_analytical
blond.legacy.blond2.impedances.impedance_sources
blond.legacy.blond2.impedances.impedance
blond.legacy.blond2.gpu
blond.legacy.blond2.beam.profile
blond.legacy.blond2.beam.beam
blond.handle_results.observables
blond.handle_results.helpers
blond.handle_results.array_recorders
blond.examples.EX_MuonCollider
blond.cycles.noise_generators.vari_noise
blond.cycles.noise_generators.base
blond.cycles.magnetic_cycle
blond.cycles.base
blond.beam_preparation.bigaussian
blond.beam_preparation.base
blond.acc_math.analytic.simple_math
blond.acc_math.analytic.hammilton
blond._generals.iterables
blond._core.simulation.simulation
blond._core.simulation.intensity_effect_manager
blond._core.ring.ring
blond._core.ring.helpers
blond._core.ring.beam_physics_relevant_elements
blond._core.helpers
blond._core.beam.particle_types
blond._core.beam.beams
blond._core.beam.base
blond._core.base
blond._core.backends.python.callables
blond._core.backends.fortran.callables
blond._core.backends.cpp.callables
blond._core.backends.backend
"""
# add type hints from database to files
for target in targets.split("\n"):
    target = target.strip()
    if len(target) == 0:
        continue
    elif target.startswith("unittests"):
        continue
    elif "legacy" in target:
        continue
    else:
        subprocess.run(f"monkeytype apply {target}".split(" "))
