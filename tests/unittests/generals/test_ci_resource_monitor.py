# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Tests for the CI resource sampler (``dev_tools/ci_resource_monitor.py``)."""

import importlib.util
import os
import sys
import tempfile

from blond.testing.backend_testing import BLonDTestCase

# Load the standalone dev_tools script by path (it is intentionally not part
# of the importable package, so it can run with a bare python3 in CI). It
# imports its sibling ``ci_omp_threads``, so dev_tools must be importable.
_DEV_TOOLS = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dev_tools"
)
sys.path.insert(0, _DEV_TOOLS)
try:
    _spec = importlib.util.spec_from_file_location(
        "ci_resource_monitor",
        os.path.join(_DEV_TOOLS, "ci_resource_monitor.py"),
    )
    ci_resource_monitor = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ci_resource_monitor)
finally:
    sys.path.remove(_DEV_TOOLS)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(text)


def _fake_cgroup(root, usage_usec=0, nr_periods=0, nr_throttled=0):
    _write(os.path.join(root, "memory.current"), "3000\n")
    _write(os.path.join(root, "memory.swap.current"), "0\n")
    _write(
        os.path.join(root, "memory.stat"),
        "anon 1000\nfile 2000\n",
    )
    _write(
        os.path.join(root, "cpu.stat"),
        f"usage_usec {usage_usec}\nuser_usec 0\nsystem_usec 0\n"
        f"nr_periods {nr_periods}\nnr_throttled {nr_throttled}\n"
        "throttled_usec 500000\n",
    )


class TestReadSample(BLonDTestCase):
    """``read_sample`` reads the job's own cgroup, not the host."""

    def test_reads_memory_and_cpu_counters(self):
        with tempfile.TemporaryDirectory() as root:
            _fake_cgroup(root, usage_usec=7, nr_periods=10, nr_throttled=2)
            sample = ci_resource_monitor.read_sample(root, now=12.5)
        self.assertEqual(sample["time"], 12.5)
        self.assertEqual(sample["memory_current"], 3000)
        self.assertEqual(sample["memory_anon"], 1000)
        self.assertEqual(sample["swap_current"], 0)
        self.assertEqual(sample["cpu_usage_usec"], 7)
        self.assertEqual(sample["nr_periods"], 10)
        self.assertEqual(sample["nr_throttled"], 2)
        self.assertEqual(sample["throttled_usec"], 500000)

    def test_missing_files_are_none(self):
        with tempfile.TemporaryDirectory() as root:
            sample = ci_resource_monitor.read_sample(root, now=0.0)
        self.assertIsNone(sample["memory_current"])
        self.assertIsNone(sample["cpu_usage_usec"])


class TestCsvRoundTrip(BLonDTestCase):
    """Samples survive being appended to and read back from the CSV."""

    def test_round_trip_keeps_missing_values(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "timeline.csv")
            first = dict.fromkeys(ci_resource_monitor.FIELDS, 1)
            second = dict.fromkeys(ci_resource_monitor.FIELDS, None)
            second["time"] = 6.0
            ci_resource_monitor.append_sample(path, first)
            ci_resource_monitor.append_sample(path, second)
            rows = ci_resource_monitor.read_samples(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["memory_current"], 1)
        self.assertEqual(rows[1]["time"], 6.0)
        self.assertIsNone(rows[1]["memory_current"])


class TestSummarize(BLonDTestCase):
    """``summarize`` turns counters into job-relative CPU/memory figures."""

    def _samples(self):
        def sample(time, usage_s, memory, periods, throttled, thr_s):
            return {
                "time": time,
                "memory_current": memory,
                "memory_anon": memory // 2,
                "swap_current": 0,
                "cpu_usage_usec": int(usage_s * 1e6),
                "nr_periods": periods,
                "nr_throttled": throttled,
                "throttled_usec": int(thr_s * 1e6),
            }

        # 0-5 s: 2 CPUs busy; 5-10 s: 8 CPUs busy (10 s and 40 s used).
        return [
            sample(0.0, 0.0, 100, 0, 0, 0.0),
            sample(5.0, 10.0, 400, 50, 5, 1.0),
            sample(10.0, 50.0, 200, 100, 30, 4.0),
        ]

    def test_cpu_is_in_cores_and_relative_to_entitlement(self):
        summary = ci_resource_monitor.summarize(
            self._samples(), cpu_entitlement=8
        )
        self.assertAlmostEqual(summary["mean_cpus"], 5.0)
        self.assertAlmostEqual(summary["peak_cpus"], 8.0)
        self.assertAlmostEqual(summary["mean_cpu_fraction"], 5.0 / 8)

    def test_memory_peaks(self):
        summary = ci_resource_monitor.summarize(
            self._samples(), cpu_entitlement=8
        )
        self.assertEqual(summary["n_samples"], 3)
        self.assertEqual(summary["peak_memory"], 400)
        self.assertEqual(summary["peak_anon"], 200)
        self.assertEqual(summary["peak_swap"], 0)

    def test_throttling(self):
        summary = ci_resource_monitor.summarize(
            self._samples(), cpu_entitlement=8
        )
        self.assertAlmostEqual(summary["throttled_seconds"], 4.0)
        self.assertAlmostEqual(summary["throttled_period_fraction"], 0.3)

    def test_single_sample_has_no_rates(self):
        summary = ci_resource_monitor.summarize(
            self._samples()[:1], cpu_entitlement=8
        )
        self.assertIsNone(summary["mean_cpus"])
        self.assertIsNone(summary["throttled_period_fraction"])

    def test_digest_is_one_line(self):
        summary = ci_resource_monitor.summarize(
            self._samples(), cpu_entitlement=8
        )
        digest = ci_resource_monitor.format_digest(summary, memory_max=1000)
        self.assertTrue(digest.startswith("resources: "))
        self.assertNotIn("\n", digest)
        self.assertIn("40%", digest)  # peak memory relative to memory.max
