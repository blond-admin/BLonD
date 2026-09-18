import unittest

import numpy as np

from blond import PooledInterpolationKick, backend
from blond.core.beam.beams import ProbeBeam
from blond.core.beam.particle_types import lead_82, proton
from blond.handle_results.helpers import callers_relative_path
from blond.testing.backend_testing import BLonDTestCase


class TestPooledInterpolationKick(BLonDTestCase):
    def setUp(self):
        self.pooled_kick = PooledInterpolationKick(maxsize=3)

    def test___init__(self):
        pass  # calls `setUp()`

    def test_clear_buffer(self):
        self.pooled_kick.register(
            time_axis=np.ones(1),
            voltage=np.ones(1),
        )
        self.pooled_kick.clear_buffer()

        self.assertEqual(len(self.pooled_kick._buffer_voltage), 0)
        self.assertEqual(len(self.pooled_kick._buffer_time_axis), 0)

    def test_register(self):
        for i in range(self.pooled_kick._maxsize + 1):  # intentional overflow
            self.pooled_kick.register(
                time_axis=np.ones(1) * i + 1,
                voltage=np.ones(1),
            )
        vals = [v[0] for v in self.pooled_kick._buffer_time_axis.values()]
        assert 0 not in vals

    def test__track(self):
        time_axis = backend.linspace(
            0,
            1,
            100,
        )
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=np.sin(time_axis),
        )
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=time_axis.copy(),
            reference_total_energy=1e12,
        )
        self.pooled_kick._track(beam=beam)
        beam_dt_copy_as_numpy_pinned = np.loadtxt(
            callers_relative_path(
                "resources/beam_dt_copy_as_numpy_pinned.npy", stacklevel=1
            )
        )

        if backend.float == np.float32:
            raise TypeError("32 bit backends have been removed.")

        np.testing.assert_allclose(
            beam.dt.copy_as_numpy()[:-1],
            beam_dt_copy_as_numpy_pinned,
            rtol=1e-12,
        )

    def test_register_and_track_with_sparse_metadata(self):
        # minimal 2-bucket sparse layout, bucket 0 and 1 both filled
        #
        # `register()` converts `time_axis`/`voltage` to the active
        # backend itself, but `sparse_metadata`'s arrays are stored
        # and forwarded as-is, so they must already be on the active
        # backend here - a plain `np.array` would fail e.g. under a
        # CUDA-active backend (leaked from another `backend_mutation`
        # test, see project convention on backend-agnostic tests).
        time_axis = backend.array([0.125, 0.375, 0.625, 0.875])
        voltage = backend.array([1.0, 2.0, 3.0, 4.0])
        sparse_metadata = {
            "first_left_cut": 0.0,
            "left_cut_distance": 0.5,
            "cut_width": 0.5,
            "bins_per_profile": 2,
            "filling_pattern": backend.array([True, True]),
            "bucket_index_to_memory_index": backend.array(
                [0, 2], dtype=np.int32
            ),
        }
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=voltage,
            sparse_metadata=sparse_metadata,
        )
        beam = ProbeBeam(
            particle_type=lead_82,
            dt=backend.array([0.125]),
            reference_total_energy=1e12,
        )
        self.pooled_kick._track(beam=beam)
        self.assertNotEqual(beam.dE.copy_as_numpy()[0], 0.0)

    def test_register_twice_overwrites_sparse_metadata(self):
        # `voltage` accumulates across repeat `register()` calls for
        # the same `time_axis`, but `sparse_metadata` describes fixed
        # geometry and should instead be overwritten by the latest
        # call, not silently retain the first call's value.
        time_axis = np.array([0.125, 0.375, 0.625, 0.875])
        voltage = np.array([1.0, 2.0, 3.0, 4.0])
        first_sparse_metadata = {
            "first_left_cut": 0.0,
            "left_cut_distance": 0.5,
            "cut_width": 0.5,
            "bins_per_profile": 2,
            "filling_pattern": np.array([True, True]),
            "bucket_index_to_memory_index": np.array([0, 2], dtype=np.int32),
        }
        second_sparse_metadata = {
            "first_left_cut": 1.0,
            "left_cut_distance": 0.5,
            "cut_width": 0.5,
            "bins_per_profile": 2,
            "filling_pattern": np.array([True, True]),
            "bucket_index_to_memory_index": np.array([0, 2], dtype=np.int32),
        }
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=voltage,
            sparse_metadata=first_sparse_metadata,
        )
        self.pooled_kick.register(
            time_axis=time_axis,
            voltage=voltage,
            sparse_metadata=second_sparse_metadata,
        )
        key = id(time_axis)
        self.assertIs(
            self.pooled_kick._buffer_sparse_metadata[key],
            second_sparse_metadata,
        )


class TestPooledKickMatchesDirectKick(BLonDTestCase):
    """Pooling must not change the physics, only when the kick is applied.

    `RFStation._track_interp` either kicks immediately or hands the kick
    to a `PooledInterpolationKick`. Both must leave the beam in the same
    state; the pool exists to save a pass over the particle arrays, not
    to approximate anything.

    Two properties are easy to lose and invisible for the common case of
    a singly-charged, co-rotating beam, so both are pinned with an ion
    and, separately, with a counter-rotating beam:

    * the kernel computes ``dE += charge * v + acceleration_kick``, so
      the reference energy change must reach it as `acceleration_kick`.
      Folding it into the voltage instead scales it by the charge.
    * the charge must carry the beam direction, since a counter-rotating
      beam sees the same field as a decelerating one.
    """

    TIME_AXIS = np.linspace(0.0, 1.0e-6, 33)
    VOLTAGE = 1.0e6 * np.sin(2 * np.pi * np.linspace(0.0, 1.0, 33))
    REFERENCE_ENERGY_CHANGE = 1.0e3

    def _probe_beam(self, particle_type, is_counter_rotating: bool):
        """Beam sampling the voltage at several points of the axis.

        Parameters
        ----------
        particle_type
            Particle species, which fixes the charge.
        is_counter_rotating
            Whether the beam travels against the reference direction.

        Returns
        -------
        ProbeBeam
            Beam with `dt` inside the time axis and `dE` at zero.
        """
        beam = ProbeBeam(
            particle_type=particle_type,
            dt=backend.array(self.TIME_AXIS[1:-1].copy(), dtype=backend.float),
            reference_total_energy=1e12,
        )
        # `ProbeBeam` does not forward `is_counter_rotating` to
        # `BeamBaseClass`, so there is no public way to build a
        # counter-rotating probe. The flag is only read back through
        # `signed_charge_with_direction`, which is what is under test.
        beam._is_counter_rotating = is_counter_rotating
        return beam

    def _direct_kick(self, particle_type, is_counter_rotating: bool):
        """`dE` after the kick applied immediately, as `_track_no_interp`.

        Parameters
        ----------
        particle_type
            Particle species, which fixes the charge.
        is_counter_rotating
            Whether the beam travels against the reference direction.

        Returns
        -------
        numpy.ndarray
            Resulting `dE`, on the host.
        """
        beam = self._probe_beam(particle_type, is_counter_rotating)
        backend.specials.kick_interpolated(
            dt=beam.read_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=backend.array(self.VOLTAGE, dtype=backend.float),
            bin_centers=backend.array(self.TIME_AXIS, dtype=backend.float),
            charge=beam.signed_charge_with_direction(),
            acceleration_kick=-self.REFERENCE_ENERGY_CHANGE,
        )
        return beam.dE.copy_as_numpy()

    def _pooled_kick(self, particle_type, is_counter_rotating: bool):
        """`dE` after the same kick routed through the pool.

        Parameters
        ----------
        particle_type
            Particle species, which fixes the charge.
        is_counter_rotating
            Whether the beam travels against the reference direction.

        Returns
        -------
        numpy.ndarray
            Resulting `dE`, on the host.
        """
        beam = self._probe_beam(particle_type, is_counter_rotating)
        pool = PooledInterpolationKick(maxsize=3)
        pool.register(
            time_axis=self.TIME_AXIS,
            voltage=self.VOLTAGE,
            reference_energy_change=self.REFERENCE_ENERGY_CHANGE,
        )
        pool._track(beam=beam)
        return beam.dE.copy_as_numpy()

    def test_matches_direct_kick_for_singly_charged_beam(self):
        np.testing.assert_allclose(
            self._pooled_kick(proton, False),
            self._direct_kick(proton, False),
            rtol=1e-12,
        )

    def test_matches_direct_kick_for_highly_charged_ion(self):
        """Acceleration must not be scaled by the charge.

        Folding the reference energy change into the registered voltage
        makes the kernel multiply it by the charge, an error of
        ``(charge - 1) * reference_energy_change`` per turn -- 81 keV a
        turn for lead, and exactly zero for protons.
        """
        np.testing.assert_allclose(
            self._pooled_kick(lead_82, False),
            self._direct_kick(lead_82, False),
            rtol=1e-12,
        )

    def test_matches_direct_kick_for_counter_rotating_beam(self):
        """The pooled charge must carry the beam direction.

        A counter-rotating beam traverses the same field in the opposite
        sense, which `signed_charge_with_direction` expresses by negating
        the charge. Using the unsigned particle charge inverts the whole
        kick.
        """
        np.testing.assert_allclose(
            self._pooled_kick(proton, True),
            self._direct_kick(proton, True),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
