# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the CI thread-count helper (``dev_tools/ci_omp_threads.py``)."""

import importlib.util
import os
import tempfile
import unittest

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so it can run with a bare python3 in CI).
_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "dev_tools",
    "ci_omp_threads.py",
)
_spec = importlib.util.spec_from_file_location("ci_omp_threads", _SCRIPT)
ci_omp_threads = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_omp_threads)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(text)


class TestCgroupCpuQuota(unittest.TestCase):
    """``cgroup_cpu_quota`` reads the container's CPU entitlement."""

    def test_cgroup_v2_quota(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu.max"), "200000 100000\n")
            self.assertEqual(ci_omp_threads.cgroup_cpu_quota(root), 2.0)

    def test_cgroup_v2_fractional_quota(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu.max"), "150000 100000\n")
            self.assertEqual(ci_omp_threads.cgroup_cpu_quota(root), 1.5)

    def test_cgroup_v2_unlimited(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu.max"), "max 100000\n")
            self.assertIsNone(ci_omp_threads.cgroup_cpu_quota(root))

    def test_cgroup_v1_quota(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu", "cpu.cfs_quota_us"), "400000\n")
            _write(os.path.join(root, "cpu", "cpu.cfs_period_us"), "100000\n")
            self.assertEqual(ci_omp_threads.cgroup_cpu_quota(root), 4.0)

    def test_cgroup_v1_unlimited(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu", "cpu.cfs_quota_us"), "-1\n")
            _write(os.path.join(root, "cpu", "cpu.cfs_period_us"), "100000\n")
            self.assertIsNone(ci_omp_threads.cgroup_cpu_quota(root))

    def test_no_cgroup_files(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(ci_omp_threads.cgroup_cpu_quota(root))

    def test_unparsable_content_is_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            _write(os.path.join(root, "cpu.max"), "nonsense\n")
            self.assertIsNone(ci_omp_threads.cgroup_cpu_quota(root))


class TestOmpNumThreads(unittest.TestCase):
    """``omp_num_threads`` combines quota, affinity and the cap."""

    def test_quota_limits_below_visible_cpus(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=2.0, max_threads=8
            ),
            2,
        )

    def test_cap_limits_below_visible_cpus(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=None, max_threads=8
            ),
            8,
        )

    def test_visible_cpus_limit_below_cap(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=4, quota=None, max_threads=8
            ),
            4,
        )

    def test_fractional_quota_rounds_down(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=2.7, max_threads=8
            ),
            2,
        )

    def test_never_less_than_one_thread(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=0.5, max_threads=8
            ),
            1,
        )

    def test_defaults_are_usable(self):
        threads = ci_omp_threads.omp_num_threads()
        self.assertGreaterEqual(threads, 1)
        self.assertLessEqual(threads, ci_omp_threads.DEFAULT_MAX_THREADS)


class TestDivide(unittest.TestCase):
    """``--divide`` shares the entitlement between concurrent processes."""

    def test_divides_between_ranks(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=8.0, max_threads=8, divide=2
            ),
            4,
        )

    def test_rounds_down_but_never_below_one(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=3.0, max_threads=8, divide=2
            ),
            1,
        )

    def test_divide_of_one_changes_nothing(self):
        self.assertEqual(
            ci_omp_threads.omp_num_threads(
                visible_cpus=64, quota=4.0, max_threads=8, divide=1
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
