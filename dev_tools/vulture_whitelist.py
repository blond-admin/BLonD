"""Names vulture must treat as used (see `[tool.vulture]` in pyproject).

Vulture matches whitelist entries by name only, not by location: ``_.foo``
marks *every* attribute or method called ``foo`` as used. Keep entries
specific and remove them once the code they excuse is gone.

Regenerate candidates with ``vulture --make-whitelist``, then file each one
under the section that explains *why* it is not dead code.
"""

# ruff: noqa
# Not real code: vulture only parses this file for the names it mentions.

# --- Called implicitly by Python -------------------------------------------
__getattr__  # module-level PEP 562 hooks (backend.py, cpp/callables.py)
exc_type  # `__exit__` protocol arguments
exc_val
exc_tb

# --- Public API or developer helpers without an internal caller ------------
_.twopi  # `backend.twopi`, documented backend constant
_.energy_lost_due_to_synchrotron_radiation_tracker  # public property
pinned_values_helper  # used by hand while writing pinned-value tests

# --- Used only by blond/experimental, which vulture does not scan ----------
populate_beam
_.freeze_wakefields
_.unfreeze_wakefields
ExperimentalFeaturesWarning

# --- Used in ways vulture cannot see ---------------------------------------
_._observe  # Simulation: pinned so `on_run_simulation` can be found
_._beams  # Simulation: same; tests pin it too
profile_time  # solvers.py: `+=` updates the array in the deque in place

# --- Required by an interface signature ------------------------------------
obs_per_turn  # `on_run_simulation` keyword, accepted but not needed

# --- Example scripts: values shown for demonstration -----------------------
_.smoothness  # EX_06 and line_density tests
_.recenter
_.plot_result_blocking
damping_time
losses
wakefield2
extraction_energy  # specifics/fccee/generate_rings.py, documents the mode

# --- Tests: mocks, fixtures and objects found by introspection -------------
_.n_turns_between_two_plots
_.cavity_tracker
_.OTFB_tracker
_.phi_rf_effective
_.energy_lost_per_turn
_.maxiter
_._animation_pause
_._animation_fignumber
_.cupy64_bit
_.numpy64_bit
_.share_of_circumference
_.problem
_.some_module
_.weird
_.programmed_cycle
_._example_densr_arr_rec
_.dummy_value
_._time
_._beta
_._section_indices_to_observe
_.sigma0
_.U0
_.common
_.not_track
_.not_found
_.to_be_found
_._make_solver
_.get_cake
_numpy_backend
simulate_BLonD3
xpart  # import checks that xsuite is installed
tilt
mock_rf_noise
test_tuple
rf_current_fine
matching
full_ring_and_rf_tracker
save_profile_b2
rf2
mac_ind
slic_ind
b_ind

# --- Suspected dead code: remove in a follow-up, then drop the entry -------
gaussian_distribution  # acc_math/analytic/simple_math.py, no caller
FloatOrArray  # cycles/magnetic_cycle.py, type alias never used
_._value_init  # MagneticCycle, written, never read
_._base_magnetic_rigidity  # MagneticCycle, written, never read
_._base_values  # MagneticCycle, "only for debugging"
_._last_section_i_observed  # observables.py, written, never read
_._n_freq  # impedances/solvers.py, written, never read
_._momentum_compaction_factor  # Ring, written, never read; a test sets it
_._beam_feedback  # RFStation, `attach_beam_feedback` stores it, never read
_._wake_pot_vals_need_update  # a test sets it, no solver reads it (stale?)
