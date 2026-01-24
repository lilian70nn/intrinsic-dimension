"""
Core functions to estimate Intrinsic Dimension (ID) from feature activations.

* **`twonn_id`**
* **`mle_id`**
* **`repeat_compute`**: A stability wrapper that runs an estimator multiple times on random data subsets 
and returns the average ID.
"""

import numpy as np
import torch
from sklearn import linear_model


def twonn_id(X, device='cuda', batch=512, fraction=0.9, verbose=False, subsample=None):

    # --- 1. Subsampling (Optional) ---
    if subsample is not None:
        N = X.shape[0]
        n_sub = int(N * subsample)
        perm = torch.randperm(N)[:n_sub]
        X = X[perm]

    # --- 2. Data Prep ---
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    X = X.double().to(device)

    N = X.shape[0]

    r1_list = []
    r2_list = []

    # --- 3. Compute Nearest Neighbors (Batchwise) ---
    with torch.no_grad():
        for i in range(0, N, batch):

            Xi = X[i:i+batch]
            Di = torch.cdist(Xi, X)
            vals, _ = torch.topk(Di, k=3, dim=1, largest=False)
            vals = vals.cpu().numpy()

            r1_list.append(vals[:, 1])
            r2_list.append(vals[:, 2])

    r1 = np.concatenate(r1_list)
    r2 = np.concatenate(r2_list)

    # --- 4. Clean Data ---
    mask_nonzero = r1 > 0
    mask_distinct = r2 > r1

    good = mask_nonzero & mask_distinct

    r1 = r1[good]
    r2 = r2[good]
    N_good = r1.shape[0]

    if verbose:
        print(f"[TwoNN] N={N}, Valid={N_good}")

    if N_good < 5:
        return 1.0

    # --- 5. TwoNN Statistics ---
    mu = np.sort(r2 / r1)

    Femp = np.arange(1, N_good + 1, dtype=np.float64) / N_good

    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    npoints = int(np.floor(N_good * fraction))

    if npoints < 2:
        return 1.0

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[:npoints, np.newaxis], y[:npoints, np.newaxis])

    d_hat = regr.coef_[0][0]

    return d_hat

def mle_id(X,device,k=10,batch=128,subsample=None,big_value=1000):

    if subsample is not None:
        N = X.shape[0]
        n_sub = int(N * subsample)
        perm = torch.randperm(N)[:n_sub]
        X = X[perm]

    X = X.to(device)
    N = X.size(0)

    out = []
    with torch.inference_mode():
        for i in range(0, N, batch):
            D = torch.cdist(X[i:i+batch], X)
            vals, _ = torch.topk(D, k=k+1, dim=1, largest=False)
            rk = vals[:, -1]
            logs = torch.log(rk.unsqueeze(1) / vals[:, 1:k])
            mi = (k - 1) / logs.sum(dim=1)
            out.append(mi.cpu())

    out = torch.cat(out)

    # --- Robust Averaging ---
    out = out[torch.isfinite(out)]
    if len(out) == 0:
        return float(big_value)

    denom = (1.0 / out).mean().item()

    if denom == 0 or not np.isfinite(denom):
        return float(big_value)

    id_est = 1.0 / denom

    if not np.isfinite(id_est):
        return float(big_value)

    return float(id_est)
    #return 1.0 / (1.0 / out).mean().item()



def repeat_compute(X,estimator,nres=3,fraction=0.9):

    """
    Bootstrap aggregation (Bagging) wrapper to improve stability.
    Runs the estimator multiple times on random subsets and averages results.
    """

    ID = []
    n = int(np.round(X.shape[0]*fraction))
    for i in range(nres):
        perm = np.random.permutation(X.shape[0])[:n]
        X_s = X[perm]
        d_hat= estimator(X_s)
        ID.append(d_hat)

    mean = np.mean(ID).item()
    #error = np.std(ID).item()
    return mean