"""GPU interop between BLonD's backend and xsuite's xobjects contexts.

Ownership rules under test:

* BLonD is the master (``WrapXsuite4Blond``): the BLonD backend decides.
  With ``backend.is_gpu`` the wrapper must run the guest line on a
  ``ContextCupy`` (building the tracker if needed) and keep all particle
  data on the device; a guest tracker already built on a mismatched
  context is refused.
* xsuite is the master (``WrapBlond4Xsuite``): the particles' context
  decides. ``ContextCupy`` particles switch the BLonD backend to cuda via
  ``setup_backend`` (with a warning); CPU particles combined with an
  explicitly chosen GPU BLonD backend are refused rather than silently
  downgraded.

These tests need a working CUDA device and are skipped otherwise.
"""

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")
xo = pytest.importorskip("xobjects")
cupy = pytest.importorskip("cupy")

try:
    _N_CUDA_DEVICES = cupy.cuda.runtime.getDeviceCount()
except Exception:  # noqa: BLE001 - any CUDA runtime error means "no GPU"
    _N_CUDA_DEVICES = 0

pytestmark = pytest.mark.skipif(
    _N_CUDA_DEVICES < 1, reason="No usable CUDA device"
)

from blond import Beam, SingleHarmonicRFStation, proton  # noqa: E402
from blond.core.backends.helpers import setup_backend  # noqa: E402
from blond.interfaces.xsuite.elements import (  # noqa: E402
    WrapBlond4Xsuite,
    WrapXsuite4Blond,
)

CIRCUMFERENCE = 26658.883
HARMONIC = 35640
VOLTAGE = 6e6
ENERGY0 = 450e9


@pytest.fixture(autouse=True)
def _restore_cpu_backend():
    """Each test starts and ends on the CPU backend (global state)."""
    setup_backend("python")
    yield
    setup_backend("python")


def _proton_p0c(total_energy):
    return float(np.sqrt(total_energy**2 - proton.mass**2))


def _headless_cavity():
    p0c = _proton_p0c(ENERGY0)
    return SingleHarmonicRFStation.headless(
        section_index=0,
        voltage=VOLTAGE,
        phi_rf=0.1,
        harmonic=HARMONIC,
        circumference=CIRCUMFERENCE,
        beam_reference_beta=p0c / ENERGY0,
        magnetic_cycle=None,
    )


def _xsuite_particles(ctx=None):
    zeta = np.array([-0.5, 0.0, 0.4])
    ptau = np.array([1e-5, 0.0, -2e-5])
    kwargs = {} if ctx is None else {"_context": ctx}
    return xt.Particles(
        p0c=_proton_p0c(ENERGY0),
        mass0=proton.mass,
        q0=proton.charge,
        zeta=zeta.copy(),
        ptau=ptau.copy(),
        **kwargs,
    )


def _blond_beam(dt, dE):
    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.asarray(dt, dtype=float).copy(),
        dE=np.asarray(dE, dtype=float).copy(),
        reference_total_energy=1e9,
    )
    return beam


# ---------------------------------------------------------------------------
# xsuite is the master: WrapBlond4Xsuite follows the particles' context.
# ---------------------------------------------------------------------------


def test_wrap_blond_follows_cupy_context_and_matches_cpu():
    """GPU particles switch BLonD to cuda; physics identical to CPU."""
    from blond import backend

    # CPU result first.
    particles_cpu = _xsuite_particles()
    WrapBlond4Xsuite(element=_headless_cavity()).track(particles_cpu)

    # GPU run: the wrapper must adopt xsuite's context (with a warning).
    particles_gpu = _xsuite_particles(ctx=xo.ContextCupy())
    wrapper = WrapBlond4Xsuite(element=_headless_cavity())
    with pytest.warns(UserWarning, match="cuda"):
        wrapper.track(particles_gpu)

    assert backend.is_gpu
    np.testing.assert_allclose(
        cupy.asnumpy(particles_gpu.zeta), particles_cpu.zeta, rtol=1e-10
    )
    np.testing.assert_allclose(
        cupy.asnumpy(particles_gpu.ptau), particles_cpu.ptau, rtol=1e-10
    )


def test_wrap_blond_follows_cupy_context_and_matches_gpu():
    """CPU particles switch BLonD to cpp/numba; physics identical to GPU."""
    from blond import backend

    # GPU result first.
    particles_gpu = _xsuite_particles(ctx=xo.ContextCupy())
    WrapBlond4Xsuite(element=_headless_cavity()).track(particles_gpu)

    # CPU run: the wrapper must adopt xsuite's context (with a warning).
    particles_cpu = _xsuite_particles()
    wrapper = WrapBlond4Xsuite(element=_headless_cavity())
    with pytest.warns(UserWarning, match="cpp"):
        wrapper.track(particles_cpu)

    assert not backend.is_gpu
    np.testing.assert_allclose(
        cupy.asnumpy(particles_gpu.zeta), particles_cpu.zeta, rtol=1e-10
    )
    np.testing.assert_allclose(
        cupy.asnumpy(particles_gpu.ptau), particles_cpu.ptau, rtol=1e-10
    )


def test_wrap_blond_gpu_losses_propagate():
    """Loss propagation works on-device (masks stay on the GPU)."""

    class _ElementThatLosesMiddleParticle:
        def track(self, beam):
            from blond.core.beam.flags import BeamFlags

            flags = beam.write_partial_flags()
            flags[1] = BeamFlags.LOST.value

    particles = _xsuite_particles(ctx=xo.ContextCupy())
    wrapper = WrapBlond4Xsuite(element=_ElementThatLosesMiddleParticle())
    with pytest.warns(UserWarning, match="cuda"):
        wrapper.track(particles)

    state = cupy.asnumpy(particles.state)
    assert state[0] > 0
    assert state[1] <= 0
    assert state[2] > 0


def test_wrap_blond_turn_keyed_schedule_works_on_gpu():
    """The at_turn-driven schedule counter also works with device arrays."""
    rf = _headless_cavity()
    voltages = VOLTAGE * np.array([1.0, 2.0, 3.0, 4.0])
    rf.schedule(attribute="voltage", value=voltages)

    ctx = xo.ContextCupy()  # tracker and particles must share one instance
    line = xt.Line(
        elements=[WrapBlond4Xsuite(element=rf)], element_names=["blond"]
    )
    line.build_tracker(_context=ctx)
    particles = _xsuite_particles(ctx=ctx)

    with pytest.warns(UserWarning, match="cuda"):
        line.track(particles, num_turns=1)
    assert rf.voltage == pytest.approx(voltages[0])
    line.track(particles, num_turns=1)
    assert rf.voltage == pytest.approx(voltages[1])


# ---------------------------------------------------------------------------
# BLonD is the master: WrapXsuite4Blond follows backend.is_gpu.
# ---------------------------------------------------------------------------


def test_wrap_xsuite_on_gpu_backend_matches_cpu():
    """With backend.is_gpu the guest runs on ContextCupy; physics identical."""
    dt = np.array([-1e-9, 0.0, 2e-9])
    dE = np.array([-1e6, 0.0, 5e6])
    length = 1.0

    def _run():
        beam = _blond_beam(dt, dE)
        line = xt.Line(elements=[xt.Drift(length=length)], element_names=["d"])
        wrapper = WrapXsuite4Blond(line)
        wrapper.track(beam)
        return beam, wrapper

    beam_cpu, _ = _run()
    dt_cpu = np.asarray(beam_cpu.read_partial_dt())
    dE_cpu = np.asarray(beam_cpu.read_partial_dE())

    setup_backend("cuda")
    beam_gpu, wrapper_gpu = _run()

    assert type(wrapper_gpu._line.tracker._context).__name__ == "ContextCupy"
    assert type(wrapper_gpu._particles.zeta).__module__ == "cupy"
    np.testing.assert_allclose(
        cupy.asnumpy(beam_gpu.dt.array_local), dt_cpu, rtol=1e-10
    )
    np.testing.assert_allclose(
        cupy.asnumpy(beam_gpu.dE.array_local), dE_cpu, rtol=1e-10
    )


def test_wrap_xsuite_gpu_losses_propagate():
    """Guest-side losses reach the BLonD flags on-device."""
    from blond.core.beam.flags import BeamFlags

    class _LineWithLostMiddleParticle(xt.Line):
        def __init__(self):
            super().__init__(
                elements=[xt.Drift(length=0.0)], element_names=["d"]
            )

        def track(self, particles, **kwargs):
            particles.state[1] = 0

    setup_backend("cuda")
    beam = _blond_beam([1e-9, 2e-9, 3e-9], [0.0, 0.0, 0.0])
    wrapper = WrapXsuite4Blond(_LineWithLostMiddleParticle())
    wrapper.track(beam)

    flags = cupy.asnumpy(beam.flags.array_local)
    assert flags[0] == BeamFlags.ACTIVE.value
    assert flags[1] == BeamFlags.LOST.value
    assert flags[2] == BeamFlags.ACTIVE.value


def test_wrap_xsuite_rejects_mismatched_tracker_context():
    """A guest tracker prebuilt on the wrong context is refused, not copied."""
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["d"])
    line.build_tracker(_context=xo.ContextCupy())

    beam = _blond_beam([0.0], [0.0])  # CPU backend (fixture default)
    wrapper = WrapXsuite4Blond(line)
    with pytest.raises(ValueError, match="context"):
        wrapper.track(beam)
