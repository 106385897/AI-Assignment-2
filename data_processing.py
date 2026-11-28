# Data processing for the traffic flow models.
# Reads the SCATS October 2006 file and turns it into arrays for LSTM/GRU.

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- settings ----
DATA_FILE = "Scats_Data_October_2006.xls"
SLOTS_PER_DAY = 96            # 24 h * 4 fifteen-minute intervals
LOOKBACK = 12                 # use the previous 12 intervals (3 hours)
HORIZON = 1                   # to predict the next interval (15 min)
TRAIN_DAYS = 25               # Oct 1-25 train, Oct 26-31 test (about 80/20)
VAL_FRACTION = 0.1            # last 10 % of the train period used for validation
MAX_GAP = 4                   # fill gaps of up to 4 intervals (1 hour) by interpolation
N_FEATURES = 3                # flow, sin(time of day), cos(time of day)

# column names V00 to V95 (one per 15 minutes)
SLOT_COLUMNS = [f"V{i:02d}" for i in range(SLOTS_PER_DAY)]


def load_raw_data(path=DATA_FILE):
    # read the Data sheet (row 0 is just a title, so header is row 1)
    raw = pd.read_excel(path, sheet_name="Data", header=1)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.normalize()   # drop the stray 00:15 time
    raw["SCATS Number"] = raw["SCATS Number"].astype(int)
    return raw


def build_site_series(raw):
    # turn each site into one long 15-minute series for the whole month

    all_days = pd.date_range(raw["Date"].min(), raw["Date"].max(), freq="D")
    series = {}

    for site, rows in raw.groupby("SCATS Number"):
        # a site has several detector rows per day, find the usual number
        rows_per_day = rows.groupby("Date").size()
        normal = rows_per_day.mode()[0]                  # usual number of detectors
        complete_days = rows_per_day[rows_per_day == normal].index

        daily = rows.groupby("Date")[SLOT_COLUMNS].sum()  # sum detectors per day
        daily = daily.reindex(all_days)                   # missing days -> NaN rows
        # days with a detector missing would give a fake low total, so mark as missing
        daily.loc[~daily.index.isin(complete_days)] = np.nan

        # flatten days x 96 columns into one timeline
        index = pd.date_range(all_days[0], periods=len(all_days) * SLOTS_PER_DAY,
                              freq="15min")
        series[site] = pd.DataFrame({"flow": daily.to_numpy(dtype=float).ravel()},
                                    index=index)
    return series


def clean_series(series, max_gap=MAX_GAP):
    # fill short gaps only, long gaps stay empty
    cleaned = {}
    for site, df in series.items():
        df = df.copy()
        is_gap = df["flow"].isna()
        gap_id = (is_gap != is_gap.shift()).cumsum()          # label each run of values
        gap_len = is_gap.groupby(gap_id).transform("sum")     # NaN-run length per row
        filled = df["flow"].interpolate(limit_area="inside")  # fill every inner gap...
        too_long = is_gap & (gap_len > max_gap)
        df["flow"] = filled.mask(too_long)                    # ...then undo the long ones
        cleaned[site] = df
    return cleaned


def time_features(index):
    # time of day as sin/cos so 23:45 and 00:00 end up close
    slot = (index.hour * 60 + index.minute) / (24 * 60)
    return np.sin(2 * np.pi * slot), np.cos(2 * np.pi * slot)



def split_by_day(df, train_days=TRAIN_DAYS, val_fraction=VAL_FRACTION):
    """
    Split one site's series into (train, val, test) by time, never shuffled.
    test = days after `train_days`; val = last part of the train period.
    """
    split_point = train_days * SLOTS_PER_DAY
    train_all, test = df.iloc[:split_point], df.iloc[split_point:]
    val_start = int(len(train_all) * (1 - val_fraction))
    return train_all.iloc[:val_start], train_all.iloc[val_start:], test

class MinMaxScaler:
    # scales values to 0-1, fitted on training data only

    def fit(self, values):
        values = values[~np.isnan(values)]                 # ignore missing values
        self.min_, self.max_ = float(values.min()), float(values.max())
        if self.max_ == self.min_:                 # avoid divide by zero
            self.max_ = self.min_ + 1.0
        return self

    def transform(self, values):
        return (values - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, values):
        # back to real vehicle counts
        return values * (self.max_ - self.min_) + self.min_


def make_windows(df, scaler, lookback=LOOKBACK, horizon=HORIZON, return_times=False):
    # slide a window over the series: X = past values, y = the next value
    # return_times=True also returns the timestamp of each target (for plots by hour)
    flow = scaler.transform(df["flow"].to_numpy(dtype=float))
    sin_t, cos_t = time_features(df.index)
    features = np.stack([flow, sin_t, cos_t], axis=1)          # (time, 3)

    X, y, times = [], [], []
    for start in range(len(df) - lookback - horizon + 1):
        window = features[start:start + lookback]
        target = flow[start + lookback + horizon - 1]
        # skip any window that has missing values
        if np.isnan(window).any() or np.isnan(target):
            continue
        X.append(window)
        y.append(target)
        times.append(df.index[start + lookback + horizon - 1])

    if not X:
        X, y = np.empty((0, lookback, N_FEATURES)), np.empty((0,))
        times = np.array([], dtype="datetime64[ns]")
    else:
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        times = np.array(times, dtype="datetime64[ns]")
    if return_times:
        return X, y, times
    return X, y


@dataclass
class Dataset:
    """Container handed to the models. See module docstring for shapes."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    site_train: np.ndarray       # site number of every sample
    site_val: np.ndarray
    site_test: np.ndarray
    scalers: dict = field(default_factory=dict)   # {site: MinMaxScaler}
    lookback: int = LOOKBACK
    t_test: np.ndarray = None    # timestamp of each test target (same order as y_test)

    def inverse_transform(self, y_scaled, sites):
        """Convert scaled values back to vehicles per 15 minutes, site by site."""
        y_scaled, sites = np.asarray(y_scaled, dtype=float).ravel(), np.asarray(sites)
        out = np.empty_like(y_scaled)
        # each site has its own scaler so do it one site at a time
        for site in np.unique(sites):
            mask = sites == site
            out[mask] = self.scalers[site].inverse_transform(y_scaled[mask])
        return np.maximum(out, 0.0)          # flow cannot be negative (float rounding)


def prepare_dataset(path=DATA_FILE, lookback=LOOKBACK, horizon=HORIZON,
                    train_days=TRAIN_DAYS, sites=None):
    """
    Full pipeline. `sites` optionally restricts to a list of SCATS numbers.
    """
    # load -> one series per site -> clean
    series = clean_series(build_site_series(load_raw_data(path)))
    if sites is not None:
        series = {s: series[s] for s in sites}

    parts = {name: ([], [], []) for name in ("train", "val", "test")}  # X, y, site
    scalers = {}
    test_times = []

    for site, df in series.items():
        train, val, test = split_by_day(df, train_days)
        # scaler learns from the training part only (no peeking at test)
        scalers[site] = MinMaxScaler().fit(train["flow"].to_numpy(dtype=float))
        for name, part in (("train", train), ("val", val), ("test", test)):
            X, y, t = make_windows(part, scalers[site], lookback, horizon,
                                   return_times=True)
            if name == "test":
                test_times.append(t)
            parts[name][0].append(X)
            parts[name][1].append(y)
            parts[name][2].append(np.full(len(y), site))

    # join all sites together
    def join(name):
        xs, ys, ss = parts[name]
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)

    X_tr, y_tr, s_tr = join("train")
    X_va, y_va, s_va = join("val")
    X_te, y_te, s_te = join("test")
    return Dataset(X_tr, y_tr, X_va, y_va, X_te, y_te, s_tr, s_va, s_te,
                   scalers, lookback, np.concatenate(test_times))


# quick check: python data_processing.py
if __name__ == "__main__":
    data = prepare_dataset()
    print("sites:", len(data.scalers))
    for name in ("train", "val", "test"):
        print(name, getattr(data, f"X_{name}").shape, getattr(data, f"y_{name}").shape)