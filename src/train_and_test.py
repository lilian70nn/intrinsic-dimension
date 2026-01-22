import copy
import torch
from tqdm import tqdm
from src.id.compute import compute_id_dynamics_across_models

def train(global_batch,
          model_dict,
          estimator_dict,
          train_loader,
          test_loader,
          criterion,
          optimizer,
          id_logging_interval,
          device,
          depth_fns=None,
          batch_scheduler=None):

    batch_log = []
    (model_name, model), = model_dict.items()
    (estimator_name, estimator), = estimator_dict.items()
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0
    model.to(device)


    for batch in train_loader:
        # Handle Tabular (3-item) vs Image (2-item) batches
        if len(batch) == 3:
            x_num, x_cat, y = batch
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            inputs = (x_num, x_cat)
        else:
            inputs, y = batch
            inputs, y = inputs.to(device), y.to(device)

        batch_size = y.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)

        # --- TabM Specific Logic (3D Output) ---
        if outputs.ndim == 3:
            B, k, C = outputs.shape
            outputs_flat = outputs.reshape(B * k, C)
            y_flat = y.repeat_interleave(k)
            loss = criterion(outputs_flat, y_flat)
            ensemble_logits = outputs.mean(dim=1)
            preds = ensemble_logits.argmax(dim=1)
        # --- Standard Logic (2D Output) ---
        else:
            loss = criterion(outputs, y)
            preds = outputs.argmax(dim=1)


        loss.backward()
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        total += batch_size
        train_loss += loss.item() * batch_size

        batch_train_acc = (preds == y).sum().item()
        train_correct += batch_train_acc

        batch_train_acc = batch_train_acc / batch_size

        batch_last_hidden_layer_id = None
        batch_test_acc = None

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
            _, batch_test_acc = test(model,test_loader,criterion,device)
            model.train()
            model.to(device)

            batch_log.append([global_batch,batch_train_acc,batch_test_acc,batch_last_hidden_layer_id])
        global_batch += 1

    epoch_train_loss = train_loss / total
    epoch_train_acc = train_correct / total
    epoch_test_loss, epoch_test_acc = test(model,test_loader,criterion,device)
    epoch_log = [epoch_train_acc,epoch_test_acc]

    return global_batch, {model_name:model} , batch_log, epoch_log



def test(model,dataloader,criterion,device):


    model.eval()
    test_loss = 0
    test_correct = 0
    total = 0
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x_num, x_cat, y = batch
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                inputs = (x_num, x_cat)
            else:
                inputs, y = batch
                inputs, y = inputs.to(device), y.to(device)

            batch_size = y.size(0)
            outputs = model(inputs)

            if outputs.ndim == 3:

                B, k, C = outputs.shape
                outputs_flat = outputs.reshape(B * k, C)
                y_flat = y.repeat_interleave(k)
                loss = criterion(outputs_flat, y_flat)

                ensemble_logits = outputs.mean(dim=1)
                preds = ensemble_logits.argmax(dim=1)

            else:

                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)


            total += batch_size
            test_loss += loss.item() * batch_size

            test_correct += (preds == y).sum().item()


    epoch_test_loss = test_loss / total
    epoch_test_acc = test_correct / total

    return epoch_test_loss, epoch_test_acc



def train_and_compute_id(
    epochs,
    model,
    train_loader,
    test_loader,
    estimator,
    criterion,
    optimizer,
    id_logging_interval,
    device,
    batch_scheduler=None,
    epoch_scheduler=None,
    depth_fns=None):



    (model_name, net), = model.items()
    result_per_batch = []
    result_per_epoch = []
    result_0_and_best_epoch = []
    global_batch = 0
    best_acc = 0
    best_model = None
    best_epoch = 0

    # 1. Compute ID for Epoch 0 (Untrained)
    id_dynamics_across_models = compute_id_dynamics_across_models(
        model,estimator,
        test_loader,
        depth_fns,
        device,pca_dim=True,
        show_progress=False)[next(iter(model))]

    result_epoch_0 = [0,test(net,test_loader,criterion,device)[1],
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
            optimizer,
            id_logging_interval,
            device,
            depth_fns=depth_fns,
            batch_scheduler=batch_scheduler)

        # Track Best Model
        if epoch_log[1] > best_acc:
            best_acc = epoch_log[1]
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
    result_0_and_best_epoch.append([best_epoch,best_acc,
                                            id_dynamics_across_models[next(iter(estimator))],
                                            id_dynamics_across_models['embdims'],
                                            id_dynamics_across_models['pca_dim']])
    return depths, result_per_batch, result_per_epoch, result_0_and_best_epoch, best_model