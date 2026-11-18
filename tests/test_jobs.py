import os
import unittest
from types import SimpleNamespace
from unittest import mock

from tools.dk.platforms.gc_mwcc import jobs


class DefaultJobsTest(unittest.TestCase):
    def test_the_setting_then_the_environment_then_the_idle_cores(self):
        cfg = SimpleNamespace(platform_options={"jobs": 3})
        with mock.patch.dict(os.environ, {"DK_JOBS": "5"}):
            self.assertEqual(jobs.default_jobs(cfg), 3)
            self.assertEqual(jobs.default_jobs(SimpleNamespace(platform_options={})), 5)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("os.cpu_count", return_value=16), \
                mock.patch("os.getloadavg", return_value=(12.4, 0, 0)):
            self.assertEqual(jobs.default_jobs(SimpleNamespace(platform_options={})), 4)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("os.cpu_count", return_value=16), \
                mock.patch("os.getloadavg", return_value=(74.0, 0, 0)):
            self.assertEqual(jobs.default_jobs(None), 1)            # an overloaded machine still gets one
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("os.cpu_count", return_value=16), \
                mock.patch("os.getloadavg", return_value=(0.0, 0, 0)):
            self.assertEqual(jobs.default_jobs(None, cap=8), 8)

    def test_an_explicit_jobs_wins(self):
        self.assertEqual(jobs.resolve(2, SimpleNamespace(platform_options={"jobs": 9})), 2)


if __name__ == "__main__":
    unittest.main()
