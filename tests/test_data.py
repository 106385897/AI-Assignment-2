# Tests for the data processing code.
# Run: python -m unittest discover -s tests -v

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import data_processing as dp  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", dp.DATA_FILE)


def toy_series(n=200, start="2006-10-01"):
    """A simple increasing series 0,1,2,... so expected windows are easy to write."""
    index = pd.date_range(start, periods=n, freq="15min")
    return pd.DataFrame({"flow": np.arange(n, dtype=float)}, index=index)


def toy_scaler():
    """Identity-like scaler: min 0, max 1 leaves values unchanged."""
    s = dp.MinMaxScaler()
    s.min_, s.max_ = 0.0, 1.0
    return s


# tests on small made-up data
class SyntheticTests(unittest.TestCase):
    # windows should have the right shape
    def test_01_window_shapes(self):
        X, y = dp.make_windows(toy_series(100), toy_scaler(), lookback=12, horizon=1)
        self.assertEqual(X.shape, (100 - 12, 12, dp.N_FEATURES))
        self.assertEqual(y.shape, (88,))

    # first window and its target are the right numbers
    def test_02_window_contents_and_target_alignment(self):
        X, y = dp.make_windows(toy_series(50), toy_scaler(), lookback=4, horizon=1)
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3])   # first window
        self.assertEqual(y[0], 4)                                  # next value
        self.assertEqual(y[-1], 49)

    # windows with missing values get skipped
    def test_03_windows_skip_nan(self):
        df = toy_series(30)
        df.iloc[10, 0] = np.nan
        X, y = dp.make_windows(df, toy_scaler(), lookback=4, horizon=1)
        self.assertFalse(np.isnan(X).any() or np.isnan(y).any())
        self.assertLess(len(y), 30 - 4)                            # some windows dropped

    # series too short for one window gives empty arrays
    def test_04_short_series_gives_empty_arrays(self):
        X, y = dp.make_windows(toy_series(5), toy_scaler(), lookback=12)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)

    # scale then unscale should give the original numbers
    def test_05_scaler_round_trip_and_range(self):
        values = np.array([10.0, 50.0, 100.0, 400.0])
        scaler = dp.MinMaxScaler().fit(values)
        scaled = scaler.transform(values)
        self.assertEqual(scaled.min(), 0.0)
        self.assertEqual(scaled.max(), 1.0)
        np.testing.assert_allclose(scaler.inverse_transform(scaled), values)

    # same value everywhere should not crash
    def test_06_scaler_constant_series_no_divide_by_zero(self):
        scaler = dp.MinMaxScaler().fit(np.array([5.0, 5.0, 5.0]))
        self.assertTrue(np.isfinite(scaler.transform(np.array([5.0]))).all())

    # missing values are ignored when fitting
    def test_07_scaler_ignores_nan_when_fitting(self):
        scaler = dp.MinMaxScaler().fit(np.array([1.0, np.nan, 9.0]))
        self.assertEqual((scaler.min_, scaler.max_), (1.0, 9.0))

    # train, val, test are in time order with no overlap
    def test_08_split_is_chronological_and_disjoint(self):
        df = toy_series(31 * dp.SLOTS_PER_DAY)
        train, val, test = dp.split_by_day(df)
        self.assertLess(train.index.max(), val.index.min())
        self.assertLess(val.index.max(), test.index.min())
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        self.assertEqual(len(test), 6 * dp.SLOTS_PER_DAY)          # Oct 26-31

    # midnight and 23:45 should look close
    def test_09_time_features_wrap_around_midnight(self):
        idx = pd.DatetimeIndex(["2006-10-01 00:00", "2006-10-01 23:45"])
        sin_t, cos_t = dp.time_features(idx)
        self.assertAlmostEqual(cos_t[0], 1.0)
        self.assertAlmostEqual(sin_t[0], 0.0)
        self.assertLess(abs(sin_t[1]), 0.1)                        # 23:45 close to 00:00

    # short gaps get filled, long gaps do not
    def test_10_interpolation_fills_short_gap_only(self):
        df = toy_series(30)
        df.iloc[5:7, 0] = np.nan          # gap of 2 -> filled
        df.iloc[15:22, 0] = np.nan        # gap of 7 -> left (max_gap = 4)
        out = dp.clean_series({1: df}, max_gap=4)[1]["flow"]
        self.assertFalse(out.iloc[5:7].isna().any())
        self.assertEqual(out.iloc[5], 5.0)                         # linear fill
        self.assertTrue(out.iloc[17:20].isna().all())
        self.assertTrue(df["flow"].iloc[5:7].isna().all())         # input untouched


# tests on the real October 2006 file
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = dp.load_raw_data(DATA_PATH)
        cls.series = dp.clean_series(dp.build_site_series(cls.raw))
        cls.data = dp.prepare_dataset(DATA_PATH)

    # the real file has 40 sites and no missing values
    def test_11_file_loads_expected_structure(self):
        self.assertEqual(self.raw["SCATS Number"].nunique(), 40)
        self.assertTrue(set(dp.SLOT_COLUMNS).issubset(self.raw.columns))
        self.assertEqual(self.raw[dp.SLOT_COLUMNS].isna().sum().sum(), 0)

    # every site covers the full month
    def test_12_every_site_has_full_31_day_index(self):
        self.assertEqual(len(self.series), 40)
        for df in self.series.values():
            self.assertEqual(len(df), 31 * dp.SLOTS_PER_DAY)
            self.assertEqual(df.index[0], pd.Timestamp("2006-10-01 00:00"))
            self.assertEqual(df.index[-1], pd.Timestamp("2006-10-31 23:45"))

    # site flow = sum of its detectors
    def test_13_site_flow_is_sum_of_detectors(self):
        site = 2200
        day = pd.Timestamp("2006-10-10")
        rows = self.raw[(self.raw["SCATS Number"] == site) & (self.raw["Date"] == day)]
        expected = rows[dp.SLOT_COLUMNS].sum().to_numpy()
        got = self.series[site].loc[day:day + pd.Timedelta(hours=23, minutes=45), "flow"]
        np.testing.assert_array_equal(got.to_numpy(), expected)

    # days with a detector missing are marked missing
    def test_14_incomplete_detector_days_are_marked_missing(self):
        # Site 3685 has several days with fewer detector rows than normal.
        counts = self.raw[self.raw["SCATS Number"] == 3685].groupby("Date").size()
        bad_days = counts[counts != counts.mode()[0]].index
        self.assertGreater(len(bad_days), 0)
        flow = self.series[3685]["flow"]
        for day in bad_days:
            self.assertTrue(flow.loc[day:day + pd.Timedelta(hours=23, minutes=45)]
                            .isna().all())

    # no negative flows
    def test_15_no_negative_flows(self):
        for df in self.series.values():
            self.assertTrue((df["flow"].dropna() >= 0).all())

    # shapes match and no NaN in the final arrays
    def test_16_dataset_arrays_are_consistent(self):
        d = self.data
        for name in ("train", "val", "test"):
            X, y, s = (getattr(d, f"X_{name}"), getattr(d, f"y_{name}"),
                       getattr(d, f"site_{name}"))
            self.assertEqual(X.shape[1:], (dp.LOOKBACK, dp.N_FEATURES))
            self.assertEqual(len(X), len(y))
            self.assertEqual(len(X), len(s))
            self.assertFalse(np.isnan(X).any() or np.isnan(y).any())

    # training targets are between 0 and 1
    def test_17_train_data_scaled_to_unit_range(self):
        self.assertGreaterEqual(self.data.y_train.min(), 0.0)
        self.assertLessEqual(self.data.y_train.max(), 1.0 + 1e-6)

    # scaler only used training data
    def test_18_scaler_fitted_on_training_period_only(self):
        # Scaler max must equal the max of the training period, not the full month.
        site = 2200
        train, _, _ = dp.split_by_day(self.series[site])
        full_max = self.series[site]["flow"].max()
        self.assertEqual(self.data.scalers[site].max_, train["flow"].max())
        self.assertLessEqual(self.data.scalers[site].max_, full_max)

    # test windows only use test data
    def test_19_no_leakage_between_train_and_test(self):
        # A test window must hold only test-period timestamps.
        site = 2200
        train, val, test = dp.split_by_day(self.series[site])
        X, y = dp.make_windows(test, self.data.scalers[site])
        # first test target is the (lookback+1)-th test value, never a train value
        expected_first = self.data.scalers[site].transform(
            test["flow"].iloc[dp.LOOKBACK])
        self.assertAlmostEqual(float(y[0]), float(expected_first), places=5)

    # converting back gives whole vehicle counts
    def test_20_inverse_transform_recovers_vehicle_counts(self):
        d = self.data
        recovered = d.inverse_transform(d.y_test[:500], d.site_test[:500])
        self.assertGreaterEqual(recovered.min(), 0)
        self.assertEqual(recovered.dtype, float)
        # recovered values must be whole vehicle counts (within float error)
        np.testing.assert_allclose(recovered, np.round(recovered), atol=1e-2)


if __name__ == "__main__":
    unittest.main()