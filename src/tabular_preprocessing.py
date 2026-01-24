"""
Orchestrating the initial preprocessing and loading of raw tabular data.
"""

import numpy as np
import pandas as pd
import torch
from src.datasets import TabularPairDataset
from torch.utils.data import DataLoader

class TabularEncoder:
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None

        self.num_fill = {}          # col -> fill_value (fit on train)
        self.cat_maps = {}          # col -> dict(value_str -> index)
        self.cat_cardinalities = [] # cardinality = number of non-missing categories (train-only)

        self._is_fitted = False

    def fit(self, X: pd.DataFrame):
        # Identify column types from training data
        self.num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numerical fill values (train-only)
        self.num_fill = {}
        for col in self.num_cols:
            s = X[col]
            fill_value = s.median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0.0
            self.num_fill[col] = float(fill_value)

        # 3 Categorical mapping (train-only), missing excluded from distinct values
        self.cat_maps = {}
        self.cat_cardinalities = []
        for col in self.cat_cols:
            s = X[col]
            non_missing = s[s.notna()].astype(str)
            categories = sorted(non_missing.unique().tolist())  # distinct values, no missing
            mapping = {v: i for i, v in enumerate(categories)}

            self.cat_maps[col] = mapping
            self.cat_cardinalities.append(len(categories))

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame):

        # Categorical encoding:
        # - Known categories (seen in training) -> indices 0 .. K-1
        # - Missing values -> -1
        # - OOV values (unseen at test time) -> -1, it will be handled explicitly inside the model

        if not self._is_fitted:
            raise RuntimeError("Call fit(X_train) before transform().")

        # Numerical
        X_num_df = X[self.num_cols].copy()
        for col in self.num_cols:
            X_num_df[col] = X_num_df[col].fillna(self.num_fill[col])

        # Categorical: missing -> -1; OOV -> -1
        X_cat_df = X[self.cat_cols].copy()
        for col in self.cat_cols:
            mapping = self.cat_maps[col]
            col_data = X_cat_df[col]

            # map known values; unknown and missing become NaN then fill -1
            encoded = col_data.astype(str).map(mapping)
            encoded = encoded.where(col_data.notna(), np.nan)   # keep missing as NaN
            encoded = encoded.fillna(-1).astype(int)

            X_cat_df[col] = encoded

        X_num = torch.tensor(X_num_df.to_numpy(), dtype=torch.float32)
        X_cat = torch.tensor(X_cat_df.to_numpy(), dtype=torch.long)
        return X_num, X_cat

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)
    

def make_dataloaders(data, num_train, num_test, seed):
    X = data.data
    y = data.target

    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:num_train+num_test]

    X_train_df = X.iloc[train_idx]
    X_test_df  = X.iloc[test_idx]

    # 1) Fit encoder on TRAIN only
    enc = TabularEncoder().fit(X_train_df)

    # 2) Transform both with same encoder
    X_train_num, X_train_cat = enc.transform(X_train_df)
    X_test_num,  X_test_cat  = enc.transform(X_test_df)

    # y encoding
    y = y.astype(str)
    classes = sorted(y.unique())
    mapping = {cls: i for i, cls in enumerate(classes)}
    y_all = y.map(mapping).to_numpy(dtype="int64")

    y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
    y_test  = torch.tensor(y_all[test_idx], dtype=torch.long)

    train_ds = TabularPairDataset(X_train_num, X_train_cat, y_train)
    test_ds  = TabularPairDataset(X_test_num, X_test_cat, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    return train_loader, test_loader, X_train_num.shape[1], enc.cat_cardinalities