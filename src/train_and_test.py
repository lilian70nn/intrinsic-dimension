"""
This section defines the low-level functions for executing a single training epoch 
and evaluating model performance.
"""

import copy
import torch
from tqdm import tqdm
from src.id.compute import compute_id_dynamics_across_models, to_device


def train(global_batch,
          model_dict,
          estimator_dict,
          train_loader,
          test_loader,
          criterion,
          metrics,
          optimizer,
          id_logging_interval,
          device,
          depth_fns=None,
          batch_scheduler=None):
    
    """
    Training loop with SUM-based loss / metric contract.

    Conventions:
        - `criterion(outputs, targets)` returns a scalar Tensor equal to the
        *sum over the batch* (not averaged).
        - `metrics(outputs, targets)` returns a scalar Tensor representing a
        *count* (e.g. number of correct predictions), not a ratio.
        - Epoch-level averages are computed by dividing accumulated sums by
        the total number of samples.

    Notes:
        - Model-specific details (e.g. ensemble outputs, k members) are handled
        entirely inside `criterion` and `metrics`.
        - The training loop is task-agnostic (classification / regression),
        assuming the above contract is satisfied.
    """

    batch_log = []
    (model_name, model), = model_dict.items()
    (estimator_name, estimator), = estimator_dict.items()
    model.train()
    train_loss = 0
    train_metrics = 0
    total = 0
    model.to(device)


    for batch in train_loader:

        inputs, y = batch
        inputs = to_device(inputs, device)
        y = y.to(device)

        batch_size = y.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        total += batch_size
        train_loss += loss.item()
        batch_train_metrics = metrics(outputs, y).item()
        train_metrics += batch_train_metrics
        batch_train_metric_mean = batch_train_metrics / batch_size

        batch_last_hidden_layer_id = None
        batch_test_metrics = None

        # Periodically log ID and Test Accuracy
        if global_batch % id_logging_interval == 0:
            batch_last_hidden_layer_id_dict = compute_id_dynamics_across_models(
                {model_name:model},
                estimator_dict,
                test_loader,
                depth_fns,
                device,
                pca_dim=False,
                only_last_hidden_layer=True,
                show_progress=False)
            batch_last_hidden_layer_id = batch_last_hidden_layer_id_dict[model_name][estimator_name][0]
            _, batch_test_metrics = test(model,test_loader,criterion,metrics,device)
            model.train()
            model.to(device)

            batch_log.append([global_batch,batch_train_metric_mean,batch_test_metrics,batch_last_hidden_layer_id])
        global_batch += 1

    epoch_train_loss = train_loss / total
    epoch_train_metrics = train_metrics / total
    epoch_test_loss, epoch_test_metrics = test(model,test_loader,criterion,metrics,device)
    epoch_log = [epoch_train_metrics,epoch_test_metrics]

    return global_batch, {model_name:model} , batch_log, epoch_log



def test(model,dataloader,criterion,metrics,device):


    model.eval()
    test_loss = 0
    test_metrics = 0
    total = 0
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:

            inputs, y = batch
            inputs = to_device(inputs, device)
            y = y.to(device)

            batch_size = y.size(0)
            outputs = model(inputs)

            total += batch_size
            test_loss += criterion(outputs, y).item()
            test_metrics += metrics(outputs, y).item()


    epoch_test_loss = test_loss / total
    epoch_test_metrics = test_metrics / total

    return epoch_test_loss, epoch_test_metrics



def train_and_compute_id(
    epochs,
    model,
    train_loader,
    test_loader,
    estimator,
    criterion,
    metrics,
    optimizer,
    id_logging_interval,
    device,
    batch_scheduler=None,
    epoch_scheduler=None,
    depth_fns=None,
    maximize=True):



    (model_name, net), = model.items()
    result_per_batch = []
    result_per_epoch = []
    result_0_and_best_epoch = []
    global_batch = 0
    best_score = -float("inf") if maximize else float("inf")
    best_model = None
    best_epoch = 0

    # 1. Compute ID for Epoch 0 (Untrained)
    id_dynamics_across_models = compute_id_dynamics_across_models(
        model,estimator,
        test_loader,
        depth_fns,
        device,pca_dim=True,
        show_progress=False)[next(iter(model))]

    result_epoch_0 = [0,test(net,test_loader,criterion,metrics,device)[1],
                      id_dynamics_across_models[next(iter(estimator))],
                      id_dynamics_across_models['embdims'],
                      id_dynamics_across_models['pca_dim'],
                      id_dynamics_across_models['depth_names'],
                      ]
    result_per_epoch.append(result_epoch_0[:3] + result_epoch_0[-1:])
    result_0_and_best_epoch.append(result_epoch_0)
    depths = id_dynamics_across_models['depths']

    # 2. Main Training Loop
    for epoch in tqdm(range(1,epochs+1),desc = "Training epochs"):
        global_batch, model, batch_log, epoch_log = train(
            global_batch,
            model,
            estimator,
            train_loader,
            test_loader,
            criterion,
            metrics,
            optimizer,
            id_logging_interval,
            device,
            depth_fns=depth_fns,
            batch_scheduler=batch_scheduler)

        # Track Best Model
        if (maximize and epoch_log[1] > best_score) or (not maximize and epoch_log[1] < best_score):
            best_score = epoch_log[1]
            best_model = copy.deepcopy(model)
            best_epoch = epoch


        result_per_batch.extend(batch_log)

        # Compute ID for current epoch
        id_dynamics_across_models = compute_id_dynamics_across_models(
                model,
                estimator,
                test_loader,
                depth_fns,
                device,pca_dim=False,
                show_progress=False)[next(iter(model))]

        epoch_id = id_dynamics_across_models[next(iter(estimator))]
        result_per_epoch.append([epoch,epoch_log[1],epoch_id])
        #print(f'epoch{epoch}, train and test acc: {epoch_log}, epoch_id:{epoch_id}')
        if epoch_scheduler is not None:
            epoch_scheduler.step()

    # 3. Final Analysis on Best Model
    id_dynamics_across_models = compute_id_dynamics_across_models(
        best_model,estimator,
        test_loader,
        depth_fns,
        device,pca_dim=True,
        show_progress=False)[next(iter(model))]
    result_0_and_best_epoch.append([best_epoch,best_score,
                                            id_dynamics_across_models[next(iter(estimator))],
                                            id_dynamics_across_models['embdims'],
                                            id_dynamics_across_models['pca_dim']])
    return depths, result_per_batch, result_per_epoch, result_0_and_best_epoch, best_model



# Loss and metric definitions.
def classif_ce_ensemble_sum(outputs, targets):
    B, k, C = outputs.shape
    outputs_flat = outputs.reshape(B * k, C)
    y_flat = targets.repeat_interleave(k)
    ce_sum = torch.nn.CrossEntropyLoss(reduction="sum")(outputs_flat, y_flat)
    return ce_sum/k

def classif_accuracy_ensemble_sum(outputs, targets):
    ensemble_logits = outputs.mean(dim=1)
    preds = ensemble_logits.argmax(dim=1)
    return (preds == targets).sum()

def regress_mse_ensemble_sum(outputs, targets):
    B, k, C = outputs.shape
    outputs_flat = outputs.reshape(B * k, C)
    y_flat = targets.repeat_interleave(k).float()
    mse_sum = torch.nn.MSELoss(reduction="sum")(outputs_flat, y_flat)
    return mse_sum/k


def accuracy_sum(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).sum()

def mse_sum(outputs, targets):
    mse = torch.nn.MSELoss(reduction="sum")(outputs, targets)
    return mse


def regress_mse_sum(outputs, targets):
    outputs = outputs.squeeze(1)
    targets = targets.float()
    mse = torch.nn.MSELoss(reduction="sum")(outputs, targets)
    return mse