import numpy as np
import torch.nn as nn

def getDepths_RealMLP_TD(model):
    modules = []
    names = []
    depths = []

    # ---- Capture checkpoints (in a fixed semantic order) ----
    cap_list = [
        ("input", model.capture_input),
        ("one-hot\nencoding", model.capture_after_ohe),
        ("scale\nclip", model.capture_after_robust),
        ("num/cat\nembeddings", model.capture_after_embed),
        ("learnable\nscaling", model.capture_after_scale),
    ]
    for i, (n, m) in enumerate(cap_list):
        modules.append(m)
        names.append(n)
        depths.append(i)

    # ---- MLP linear layers (4 linears) ----
    linear_layers = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
    for j, layer in enumerate(linear_layers):
        modules.append(layer)
        names.append("out" if j == len(linear_layers) - 1 else f"fc{j+1}")
        depths.append(len(depths))

    return modules, names, np.array(depths)



def getDepths_StandardMLP(model):
    modules = []
    names = []
    depths = []

    # ---- Capture checkpoints (in a fixed semantic order) ----
    cap_list = [
        ("input", model.capture_input),
        ("preprocessing", model.capture_after_preprocess),
    ]
    for i, (n, m) in enumerate(cap_list):
        modules.append(m)
        names.append(n)
        depths.append(i)

    # ---- MLP linear layers (4 linears) ----
    linear_layers = [layer for layer in model.mlp if isinstance(layer, nn.Linear)]
    for j, layer in enumerate(linear_layers):
        modules.append(layer)
        names.append("out" if j == len(linear_layers) - 1 else f"fc{j+1}")
        depths.append(len(depths))

    return modules, names, np.array(depths)



def getDepths_TabM(model):
    modules = []
    names = []
    depths = []

    # 1) Capture checkpoints
    cap_list = [
        ("input", model.capture_input),
        ("preprocessing", model.capture_after_preprocess),
    ]
    for n, m in cap_list:
        modules.append(m)
        names.append(n)
        depths.append(len(depths))

    # 2) BatchEnsemble blocks + final ensemble head
    #    Keep order as in model.ensemble_mlp (Sequential)
    for layer in model.ensemble_mlp:
        cls_name = layer.__class__.__name__
        if cls_name in ("LinearBatchEnsemble", "LinearEnsemble"):
            modules.append(layer)
            if cls_name == "LinearEnsemble":
                names.append("out")
            else:
                # count fc blocks in the order they appear
                names.append(f"be_fc{sum(1 for x in names if x.startswith('be_fc')) + 1}")
            depths.append(len(depths))

    return modules, names, np.array(depths)