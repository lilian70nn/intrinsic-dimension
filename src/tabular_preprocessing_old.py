import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def encode_tabular(X,y):

    """
    Basic cleaning and encoding for raw tabular data (Pandas DataFrame and Series).

    The function performs:
        1. Identification of Numerical and Categorical columns.
        2. Numerical Imputation: Missing values are filled with the Mode.
        3. Categorical Encoding: Missing values are assigned a 'missing' token, and columns are then Label-Encoded.
        4. Target Encoding: Labels (y) are encoded into contiguous integers.

    Returns:
        X_num (torch.Tensor): Tensor of preprocessed numerical features (float32).
        X_cat (torch.Tensor): Tensor of preprocessed categorical features (long).
        y_encoded (np.ndarray): Encoded target labels (int64).
        cat_counts (list[int]): List of cardinalities (number of unique classes) for each categorical feature.
    """

    # 1. Identify column types
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    # 2. Process Numerical Columns
    X_num_df = X[num_cols].copy()
    for col in num_cols:
        mode = X_num_df[col].mode(dropna=True)
        if len(mode) == 0:
            fill_value = 0
        else:
            fill_value = mode[0]
        X_num_df[col] = X_num_df[col].fillna(fill_value)

    # 3. Process Categorical Columns
    X_cat_df = X[cat_cols].copy()
    cat_counts = []
    for col in cat_cols:
        s = X_cat_df[col].astype(str)
        s = s.mask(X_cat_df[col].isna(), "missing")
        cat_series = s.astype("category")
        X_cat_df[col] = cat_series.cat.codes
        cat_counts.append(len(cat_series.cat.categories))

    # 4. Convert to Tensors
    X_num = torch.tensor(X_num_df.to_numpy(), dtype=torch.float32)
    X_cat = torch.tensor(X_cat_df.to_numpy(), dtype=torch.long)


    # 5. Process Target Labels
    y = y.astype(str)
    classes = sorted(y.unique())
    mapping = {cls: i for i, cls in enumerate(classes)}
    y_encoded = y.map(mapping).to_numpy(dtype="int64")

    return X_num,X_cat,y_encoded,cat_counts



def make_dataloaders(data,num_train,num_test,seed):

    """
    Orchestrates the data split and final packaging into PyTorch DataLoaders.

    Args:
        data: OpenML dataset object (containing .data and .target).
        num_train (int): Number of samples for the training set.
        num_test (int): Number of samples for the testing set.
        seed (int): Random seed for splitting the data.

    Returns:
        train_loader, test_loader: PyTorch DataLoaders.
        num_numerical_features (int): The final count of numerical features.
        cat_cardinalities (list[int]): Cardinalities of all categorical features.
    """

    X = data.data
    y = data.target

    X_num, X_cat, y_all, cat_card = encode_tabular(X, y)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    train_idx = indices[:num_train]
    test_idx = indices[num_train:num_train+num_test]

    X_train_num = X_num[train_idx]
    X_train_cat = X_cat[train_idx]
    y_train = torch.tensor(y_all[train_idx], dtype=torch.long)

    X_test_num  = X_num[test_idx]
    X_test_cat  = X_cat[test_idx]
    y_test = torch.tensor(y_all[test_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train_num, X_train_cat, y_train)
    test_ds  = TensorDataset(X_test_num, X_test_cat, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    return train_loader,test_loader,X_train_num.shape[1],cat_card