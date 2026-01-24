"""
This is the core analysis engine of the experiment. It orchestrates the extraction of hidden representations 
from deep neural networks and computes their geometric properties (Intrinsic Dimension and PCA-based dimensionality).
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

th = 0.9

def get_pca_dim(x, th):
    cs = np.cumsum(x)
    indices = np.argwhere(cs > th)

    if indices.size > 0:
        return indices[0][0].item()
    else:
        return 0
    
def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def compute_id_dynamics_across_models(models, estimators, dataloader, depth_fns,
                                      device, pca_dim=False,
                                      only_last_hidden_layer=False,
                                      show_progress=True,
                                      ):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if show_progress:
        iterator = tqdm(models.items(), desc="Models")
    else:
        iterator = models.items()

    id_all_model = {}
    for model_name, model in iterator:

        model = model.to(device)
        model = model.eval()

        # 1. Determine layers to hook (modules) and their depths

        modules, names, depths = depth_fns[model_name](model)
        id_all_model[model_name] = {}
        id_all_model[model_name]['depths'] = depths
        id_all_model[model_name]['depth_names'] = names
        # If the first entry is the special "input" marker, we treat the model input
        # itself as a layer and collect its activations manually (no forward hook).
        has_input = (len(modules) > 0 and isinstance(modules[0], str) and modules[0] == "input")

        if has_input:
            modules = modules[1:]

        # 2. Filter for only last hidden layer if requested
        if only_last_hidden_layer:
            modules = modules[-2:-1]
            names = names[-2:-1]
            depths = depths[-2:-1]


        # 3. Register hooks for all checkpoints (single forward pass per batch)
        # Unlike a layer-by-layer approach (one forward per layer), we register hooks once and
        # obtain all layer activations in a single forward pass per batch.
        n_layers = len(modules)
        activations_all_layer = [[] for _ in range(n_layers)]

        def make_hook(layer_idx):
            def hook(module, input, output):
                if output.ndim == 3:
                    # TabM-style ensemble output: treat k members as independent samples
                    B, k, D = output.shape
                    output = output.reshape(B * k, D)
                output = output.reshape(output.shape[0], -1)
                activations_all_layer[layer_idx].append(
                    output.detach().cpu()
                )
            return hook

        handles = []
        for i, module in enumerate(modules):
            h = module.register_forward_hook(make_hook(i))
            handles.append(h)


        # 4. One pass over dataloader to collect activations
        activations_input = []
        with torch.no_grad():
            for k, batch in enumerate(dataloader, 0):

                inputs, _ = batch
                inputs = to_device(inputs, device)

                if has_input and not only_last_hidden_layer:
                    activations_input.append(inputs.reshape(inputs.shape[0], -1).detach().cpu())

                _ = model(inputs)

        for h in handles:
            h.remove()


        # 5. store results
        activations_all_layer = [
            torch.cat(layer_acts, dim=0)
            for layer_acts in activations_all_layer
        ]

        if has_input and not only_last_hidden_layer:
            X_in = torch.cat(activations_input, dim=0)
            activations_all_layer = [X_in] + activations_all_layer

        for estimator_name, estimator_fun in estimators.items():
            ids_all_layer = [estimator_fun(X) for X in activations_all_layer]
            id_all_model[model_name][estimator_name] = ids_all_layer

        if pca_dim:

            pca_all_layer = []
            for X in activations_all_layer:
                scaler = StandardScaler()
                Xn = scaler.fit_transform(X.numpy() if torch.is_tensor(X) else X)
                pca = PCA()
                pca.fit(Xn)
                pca_all_layer.append(get_pca_dim(pca.explained_variance_ratio_, th))

            id_all_model[model_name]['embdims'] = [X.shape[1] for X in activations_all_layer]
            id_all_model[model_name]['pca_dim'] = pca_all_layer


        model.to("cpu")
        torch.cuda.empty_cache()

    return id_all_model


