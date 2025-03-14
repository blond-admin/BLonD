"""
@author: Konstantinos Iliakis, George Tsapatsaris
"""

from __future__ import annotations
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from scipy.constants import c

from blond.trackers.utilities import hamiltonian
from . import GPU_DEV
from ..utils import precision
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import NDArray
    from cupy.typing import NDArray as CupyNDArray

    from ..beam.beam import Beam
    from ..input_parameters.rf_parameters import RFStation
    from ..input_parameters.ring import Ring
    from ..utils.types import SolverTypes



def rf_volt_comp(voltage: CupyNDArray, omega_rf: CupyNDArray,
                 phi_rf: CupyNDArray, bin_centers: CupyNDArray):
    """Calculate the rf voltage at each profile bin

    Args:
        voltage (float array): _description_
        omega_rf (float array): _description_
        phi_rf (float array): _description_
        bin_centers (float array): _description_

    Returns:
        float array: the calculated rf_voltage
    """

    rf_volt_comp_kernel = GPU_DEV.mod.get_function("rf_volt_comp")

    assert voltage.dtype == precision.real_t
    assert omega_rf.dtype == precision.real_t
    assert phi_rf.dtype == precision.real_t
    assert bin_centers.dtype == precision.real_t

    rf_voltage = cp.zeros(bin_centers.size, precision.real_t)

    rf_volt_comp_kernel(args=(voltage, omega_rf, phi_rf, bin_centers,
                              np.int32(voltage.size), np.int32(bin_centers.size), rf_voltage),
                        block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)
    return rf_voltage


def kick(dt: CupyNDArray, dE: CupyNDArray, voltage: CupyNDArray,
         omega_rf: CupyNDArray, phi_rf: CupyNDArray,
         charge:float, n_rf:int, acceleration_kick:float):
    """Apply the energy kick

    Args:
        dt (float array): the time coordinate
        dE (float array): the energy coordinate
        voltage (float array): _description_
        omega_rf (float array): _description_
        phi_rf (float array): _description_
        charge (float): _description_
        n_rf (int): _description_
        acceleration_kick (float): _description_
    """
    kick_kernel = GPU_DEV.mod.get_function("simple_kick")


    if not (voltage.flags.f_contiguous or voltage.flags.c_contiguous):
        warnings.warn("voltage must be contigous!")
        voltage = voltage.astype(dtype=precision.real_t, order='C', copy=False)
    if not (omega_rf.flags.f_contiguous or omega_rf.flags.c_contiguous):
        warnings.warn("omega_rf must be contigous!")
        omega_rf = omega_rf.astype(dtype=precision.real_t, order='C', copy=False)
    if not (phi_rf.flags.f_contiguous or phi_rf.flags.c_contiguous):
        warnings.warn("phi_rf must be contigous!")
        phi_rf = phi_rf.astype(dtype=precision.real_t, order='C', copy=False)

    assert isinstance(dt, cp.ndarray)
    assert isinstance(dE, cp.ndarray)
    assert isinstance(voltage, cp.ndarray)
    assert isinstance(omega_rf, cp.ndarray)
    assert isinstance(phi_rf, cp.ndarray)

    kick_kernel(args=(dt,
                      dE,
                      np.int32(n_rf),
                      precision.real_t(charge),
                      voltage,
                      omega_rf,
                      phi_rf,
                      np.int32(dt.size),
                      precision.real_t(acceleration_kick),
                      ),
                block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)

def losses_longitudinal_cut(dt:CupyNDArray, id:CupyNDArray, dt_min:float,
                            dt_max:float):  # todo testcase

    losses_longitudinal_cut_kernel = GPU_DEV.mod.get_function(
        "losses_longitudinal_cut"
    )

    losses_longitudinal_cut_kernel(
        args=(
            dt,
            id,
            len(dt),
            dt_min,
            dt_max,
        ),
        block=GPU_DEV.block_size,
        grid=GPU_DEV.grid_size,
    )
def losses_energy_cut(dE: CupyNDArray, id: CupyNDArray, dE_min:float,
                      dE_max:float):
    # todo testcase
    losses_energy_cut_kernel = GPU_DEV.mod.get_function(
        "losses_energy_cut"
    )

    losses_energy_cut_kernel(
        args=(
            dE,
            id,
            len(dE),
            dE_min,
            dE_max,
        ),
        block=GPU_DEV.block_size,
        grid=GPU_DEV.grid_size,
    )
def losses_below_energy(dE:CupyNDArray, id:CupyNDArray, dE_min:float):
    # todo testcase
    losses_below_energy_kernel = GPU_DEV.mod.get_function(
        "losses_below_energy"
    )

    losses_below_energy_kernel(
        args=(
            dE,
            id,
            len(dE),
            dE_min,
        ),
        block=GPU_DEV.block_size,
        grid=GPU_DEV.grid_size,
    )

@handle_legacy_kwargs
def losses_separatrix(
    ring: Ring,
    rf_station: RFStation,
    beam: Beam,
    dt: CupyNDArray,
    dE: CupyNDArray,
    id: CupyNDArray,
    total_voltage: Optional[NDArray] = None,
) -> None:
    r"""Function checking whether coordinate pair(s) are inside the separatrix.
    Uses the single-RF sinusoidal Hamiltonian.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    beam : class
        A Beam type class
    dt : float array
        Time coordinates of the particles to be checked
    dE : float array
        Energy coordinates of the particles to be checked
    total_voltage : float array
        Total voltage to be used if not single-harmonic RF

    Returns
    -------
    bool array
        True/False array for the given coordinates

    """
    if total_voltage is not None:
        raise NotImplementedError
    warnings.filterwarnings("once")

    if ring.n_sections > 1:
        warnings.warn(
            "WARNING: in is_in_separatrix(): the usage of several"
            + " sections is not yet implemented!"
        )
    if rf_station.n_rf > 1:
        warnings.warn(
            "WARNING in is_in_separatrix(): taking into account"
            + " the first harmonic only!"
        )

    counter = rf_station.counter[0]
    dt_sep = (
        np.pi - rf_station.phi_s[counter] - rf_station.phi_rf_d[0, counter]
    ) / rf_station.omega_rf[0, counter]

    hamilton_separation = hamiltonian(
        ring=ring,
        rf_station=rf_station,
        beam=beam,
        dt=dt_sep,
        dE=0,
    )

    warnings.filterwarnings("once")

    if ring.n_sections > 1:
        warnings.warn(
            "WARNING: The Hamiltonian is not yet properly computed for several sections!"
        )
    if rf_station.n_rf > 1:
        warnings.warn(
            "WARNING: The Hamiltonian will be calculated for the first harmonic only!"
        )
    ######################################################################
    # hamiltonian calculation, but rewritten without attribute access
    # to be executable on GPU
    counter = rf_station.counter[0]
    h0 = float(rf_station.harmonic[0, counter])
    V0 = float(rf_station.voltage[0, counter]) * rf_station.particle.charge

    # slippage factor as a function of the energy offset
    slippage_by_energy = float(
        rf_station.eta_tracking(beam, counter, dE)
    )  # todo will fail if self.alpha_order != 0
    # if
    ring_circumference = ring.ring_circumference
    beam_beta = beam.beta
    beam_energy = beam.energy
    phi_s_turn_i = rf_station.phi_s[counter]
    phi_rf_turn_i = rf_station.omega_rf[0, counter]
    phi_rf_d_turn_i = rf_station.phi_rf_d[0, counter]
    eta0_turn_i = rf_station.eta_0[counter]

    c1 = (
        slippage_by_energy
        * c
        * np.pi
        / (ring_circumference * beam_beta * beam_energy)
    )
    c2 = c * beam_beta * V0 / (h0 * ring_circumference)
    ######################################################################
    eliminate_particles_with_hamiltonian_kernel = GPU_DEV.mod.get_function(
        "eliminate_particles_with_hamiltonian"
    )
    eliminate_particles_with_hamiltonian_kernel(
        block=GPU_DEV.block_size,
        grid=GPU_DEV.grid_size,
        args=(
            precision.real_t(hamilton_separation),
            precision.real_t(c1),
            precision.real_t(c2),
            dE,
            dt,
            precision.real_t(eta0_turn_i),
            id,
            precision.real_t(phi_rf_d_turn_i),
            precision.real_t(phi_rf_turn_i),
            precision.real_t(phi_s_turn_i),
            len(dE),
        ),
    )


def drift(dt: CupyNDArray, dE: CupyNDArray, solver: str, t_rev: float,
          length_ratio: float, alpha_order: float, eta_0: float,
          eta_1: float, eta_2: float, alpha_0: float, alpha_1: float,
          alpha_2: float, beta: float, energy: float):
    """Apply the time drift function.

    Args:
        dt (_type_): _description_
        dE (_type_): _description_
        solver (_type_): _description_
        t_rev (_type_): _description_
        length_ratio (_type_): _description_
        alpha_order (_type_): _description_
        eta_0 (_type_): _description_
        eta_1 (_type_): _description_
        eta_2 (_type_): _description_
        alpha_0 (_type_): _description_
        alpha_1 (_type_): _description_
        alpha_2 (_type_): _description_
        beta (_type_): _description_
        energy (_type_): _description_
    """
    drift_kernel = GPU_DEV.mod.get_function("drift")

    solver_to_int = {
        'simple': 0,
        'legacy': 1,
        'exact': 2,
    }
    solver = solver_to_int[solver]

    if not isinstance(t_rev, precision.real_t):  # todo bugfix typecheck for cupy type
        t_rev = precision.real_t(
            t_rev)  # todo in order for this line to work, we need .get() find out python versioning

    drift_kernel(args=(dt, dE, solver,
                       precision.real_t(t_rev), precision.real_t(length_ratio),
                       precision.real_t(alpha_order), precision.real_t(eta_0),
                       precision.real_t(eta_1), precision.real_t(eta_2),
                       precision.real_t(alpha_0), precision.real_t(alpha_1),
                       precision.real_t(alpha_2),
                       precision.real_t(beta), precision.real_t(energy),
                       np.int32(dt.size)),
                 block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


def linear_interp_kick(dt: CupyNDArray, dE: CupyNDArray, voltage: CupyNDArray,
                       bin_centers: CupyNDArray, charge: float,
                       acceleration_kick: float):
    """An accelerated version of the kick function.

    Args:
        dt (_type_): _description_
        dE (_type_): _description_
        voltage (_type_): _description_
        bin_centers (_type_): _description_
        charge (_type_): _description_
        acceleration_kick (_type_): _description_
    """
    gm_linear_interp_kick_help = GPU_DEV.mod.get_function("lik_only_gm_copy")
    gm_linear_interp_kick_comp = GPU_DEV.mod.get_function("lik_only_gm_comp")

    assert dt.dtype == precision.real_t
    assert dE.dtype == precision.real_t
    assert voltage.dtype == precision.real_t
    assert isinstance(voltage, cp.ndarray)
    assert bin_centers.dtype == precision.real_t
    assert isinstance(bin_centers, cp.ndarray)

    macros = dt.size
    slices = bin_centers.size

    glob_vkick_factor = cp.empty(2 * (slices - 1), precision.real_t)
    gm_linear_interp_kick_help(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=GPU_DEV.grid_size, block=GPU_DEV.block_size)

    gm_linear_interp_kick_comp(args=(dt,
                                     dE,
                                     voltage,
                                     bin_centers,
                                     precision.real_t(charge),
                                     np.int32(slices),
                                     np.int32(macros),
                                     precision.real_t(acceleration_kick),
                                     glob_vkick_factor),
                               grid=GPU_DEV.grid_size, block=GPU_DEV.block_size)


def slice_beam(dt: CupyNDArray, profile: CupyNDArray,
               cut_left: float, cut_right: float):
    """Constant space slicing with a constant frame.

    Args:
        dt (_type_): _description_
        profile (_type_): _description_
        cut_left (_type_): _description_
        cut_right (_type_): _description_
    """
    sm_histogram = GPU_DEV.mod.get_function("sm_histogram")
    hybrid_histogram = GPU_DEV.mod.get_function("hybrid_histogram")

    assert dt.dtype == precision.real_t, f"{dt.dtype=}, not {precision.real_t}"

    n_slices = profile.size
    profile.fill(0)

    if not isinstance(cut_left, float):
        cut_left = float(cut_left)
    if not isinstance(cut_right, float):
        cut_right = float(cut_right)

    if 4 * n_slices < GPU_DEV.attributes['MaxSharedMemoryPerBlock']:
        sm_histogram(args=(dt, profile, precision.real_t(cut_left),
                           precision.real_t(cut_right), np.uint32(n_slices),
                           np.uint32(dt.size)),
                     grid=GPU_DEV.grid_size, block=GPU_DEV.block_size, shared_mem=4 * n_slices)
    else:
        hybrid_histogram(args=(dt, profile, precision.real_t(cut_left),
                               precision.real_t(cut_right), np.uint32(n_slices),
                               np.uint32(dt.size), np.int32(
            GPU_DEV.attributes['MaxSharedMemoryPerBlock'] / 4)),
                         grid=GPU_DEV.grid_size, block=GPU_DEV.block_size,
                         shared_mem=GPU_DEV.attributes['MaxSharedMemoryPerBlock'])


def synchrotron_radiation(dE: CupyNDArray, U0: float, n_kicks: int,
                          tau_z: float):
    """Track particles with SR only (without quantum excitation)

    Args:
        dE (_type_): _description_
        U0 (_type_): _description_
        n_kicks (_type_): _description_
        tau_z (_type_): _description_
    """
    synch_rad = GPU_DEV.mod.get_function("synchrotron_radiation")

    assert dE.dtype == precision.real_t

    synch_rad(args=(dE, precision.real_t(U0 / n_kicks), np.int32(dE.size),
                    precision.real_t(tau_z * n_kicks),
                    np.int32(n_kicks)),
              block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


def synchrotron_radiation_full(dE: CupyNDArray, U0: float, n_kicks: int,
                               tau_z: float, sigma_dE: float, energy: float):
    """Track particles with SR and quantum excitation.

    Args:
        dE (_type_): _description_
        U0 (_type_): _description_
        n_kicks (_type_): _description_
        tau_z (_type_): _description_
        sigma_dE (_type_): _description_
        energy (_type_): _description_
    """
    synch_rad_full = GPU_DEV.mod.get_function("synchrotron_radiation_full")

    assert dE.dtype == precision.real_t

    synch_rad_full(args=(dE, precision.real_t(U0 / n_kicks), np.int32(dE.size),
                         precision.real_t(sigma_dE),
                         precision.real_t(tau_z * n_kicks),
                         precision.real_t(energy), np.int32(n_kicks)),
                   block=GPU_DEV.block_size, grid=GPU_DEV.grid_size)


@cp.fuse(kernel_name='beam_phase_helper')
def __beam_phase_helper(bin_centers: CupyNDArray, profile: CupyNDArray,
                        alpha: float, omega_rf: float, phi_rf: float):
    """Helper function, used by beam_phase

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        alpha (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_

    Returns:
        _type_: _description_
    """
    base = cp.exp(alpha * bin_centers) * profile
    a = omega_rf * bin_centers + phi_rf
    return base * cp.sin(a), base * cp.cos(a)


def beam_phase(bin_centers: NDArray, profile: NDArray, alpha: float,
               omega_rf: float, phi_rf: float, bin_size: float):
    """Beam phase measured at the main RF frequency and phase. The beam is
       convolved with the window function of the band-pass filter of the
       machine. The coefficients of sine and cosine components determine the
       beam phase, projected to the range -Pi/2 to 3/2 Pi. Note that this beam
       phase is already w.r.t. the instantaneous RF phase.*

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        alpha (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_
        bin_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert bin_centers.dtype == precision.real_t
    assert profile.dtype == precision.real_t

    array1, array2 = __beam_phase_helper(
        bin_centers, profile, alpha, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)


@cp.fuse(kernel_name='beam_phase_fast_helper')
def __beam_phase_fast_helper(bin_centers: CupyNDArray, profile: CupyNDArray,
                             omega_rf: float, phi_rf: float):
    """Helper function used by beam_phase_fast

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_

    Returns:
        _type_: _description_
    """
    arr = omega_rf * bin_centers + phi_rf
    return profile * cp.sin(arr), profile * cp.cos(arr)


def beam_phase_fast(bin_centers: CupyNDArray, profile: CupyNDArray,
                    omega_rf: float, phi_rf: float, bin_size: float):
    """Simplified, faster variation of the beam_phase function

    Args:
        bin_centers (_type_): _description_
        profile (_type_): _description_
        omega_rf (_type_): _description_
        phi_rf (_type_): _description_
        bin_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert bin_centers.dtype == precision.real_t
    assert profile.dtype == precision.real_t

    array1, array2 = __beam_phase_fast_helper(
        bin_centers, profile, omega_rf, phi_rf)
    # due to the division, the bin_size is not needed
    scoeff = cp.trapz(array1, dx=1)
    ccoeff = cp.trapz(array2, dx=1)

    return float(scoeff / ccoeff)


def kickdrift_considering_periodicity(
    acceleration_kick: float,
    beam_dE: CupyNDArray,
    beam_dt: CupyNDArray,
    rf_station: RFStation,
    solver: SolverTypes,
    turn: int,
):
    if solver != "simple":
        msg = (
            "If you require faster tracking with 'periodicity=True' on the GPU:"
            " Switch solver to 'simple' or rewrite"
            " 'kickdrift_considering_periodicity' to consider "
            "different drift equation!"
        )
        raise Exception(msg)
    # parameters to calculate coeff of drift
    # Coeff is precalulated on GPU, to spare the time on the GPU
    T0 = rf_station.t_rev[turn + 1]
    length_ratio = rf_station.length_ratio
    eta_zero = rf_station.eta_0[turn + 1]
    beta = rf_station.beta[turn + 1]
    energy = rf_station.energy[turn + 1]
    n_rf = rf_station.voltage.shape[0]
    kickdrift_considering_periodicity = GPU_DEV.mod.get_function(
        "kickdrift_considering_periodicity"
    )
    assert beam_dt.dtype == precision.real_t
    assert beam_dE.dtype == precision.real_t

    voltage = rf_station.voltage[:, turn]
    omega_rf = rf_station.omega_rf[:, turn]
    phi_rf = rf_station.phi_rf[:, turn]

    assert voltage.data.contiguous
    assert omega_rf.data.contiguous
    assert phi_rf.data.contiguous

    assert isinstance(beam_dt, cp.ndarray)
    assert isinstance(beam_dE, cp.ndarray)

    if not isinstance(voltage, cp.ndarray):
        voltage = cp.array(voltage, dtype=precision.real_t)
    if not isinstance(omega_rf, cp.ndarray):
        omega_rf = cp.array(omega_rf, dtype=precision.real_t)
    if not isinstance(phi_rf, cp.ndarray):
        phi_rf = cp.array(phi_rf, dtype=precision.real_t)



    kickdrift_considering_periodicity(
        args=(
            beam_dt,
            beam_dE,
            precision.real_t(rf_station.t_rev[turn + 1]),
            n_rf,
            voltage,
            omega_rf,
            phi_rf,
            precision.real_t(rf_station.particle.charge),
            precision.real_t(acceleration_kick),
            precision.real_t(
                T0 * length_ratio * eta_zero / (beta * beta * energy)
            ),
            # turn+1
            len(beam_dt),
        ),
        block=GPU_DEV.block_size,
        grid=GPU_DEV.grid_size,
    )
