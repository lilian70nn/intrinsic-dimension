from functools import partial

import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml

from src.id.estimator import twonn_id
from src.id.compute import compute_id_dynamics_across_models
from src.train_and_test_old import train_and_compute_id
from src.plots import plot_fig3b, plot_fig5c, plot_fig9A, plot_fig9B, plot_fig9C
from src.tabular_preprocessing import make_dataloaders
from src.tabular_depths import getDepths_RealMLP_TD, getDepths_StandardMLP, getDepths_TabM
from src.tabular_models import RealMLP_TD, StandardMLP, TabM
from src.train_and_test import classif_ce_ensemble_sum, classif_accuracy_ensemble_sum, regress_mse_ensemble_sum, accuracy_sum,regress_mse_sum

import argparse
from pathlib import Path

def adult_id_full_experiment(show=True,savepath=None,device=None,
            epochs=15,
            id_logging_interval=15):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Configuration Parameters ---
    num_train = 20000
    num_test = 2000
    adult = fetch_openml(data_id=1590, as_frame=True)
    adult_train,adult_test, adult_num_numerical, adult_cardinality = make_dataloaders(adult,num_train,num_test,seed=42)
    
    RealMLP_adult={"RealMLP":RealMLP_TD(adult_num_numerical,adult_cardinality,num_classes=2)}
    RealMLP_adult["RealMLP"].fit_statistics(
        adult_train.dataset.x_num,
        adult_train.dataset.x_cat
    )

    StandardMLP_adult={"StandardMLP":StandardMLP(adult_num_numerical,adult_cardinality,num_classes=2)}
    StandardMLP_adult["StandardMLP"].fit_statistics(adult_train.dataset.x_num)

    depth_fns = {
         "RealMLP":getDepths_RealMLP_TD,
         "StandardMLP":getDepths_StandardMLP,
         "TabM":getDepths_TabM
    }

    estimator = {"TwoNN": partial(twonn_id,device=device,batch=512)}
            
    criterion = nn.CrossEntropyLoss()
    metrics = accuracy_sum

    # --- ID Measurement of Untrained Models--
    id_dynamics_realmlp_untrained = compute_id_dynamics_across_models(
      RealMLP_adult,
      estimator,
      adult_test,
      depth_fns,
      device,
      pca_dim=False)

    id_dynamics_standardmlp_untrained = compute_id_dynamics_across_models(
        StandardMLP_adult,
        estimator,
        adult_test,
        depth_fns,
        device,
        pca_dim=False)

    id_dynamics_untrained = {**id_dynamics_realmlp_untrained,
                            **id_dynamics_standardmlp_untrained}

    title = "Intrinsic Dimension on Adult Before Training (TwoNN)"
    plot_fig3b(id_dynamics_untrained,"TwoNN",title,annotate=True,show=show,savepath=savepath)


    # --- Evaluation of RealMLP on Adult ---
    # epochs = 15
    # id_logging_interval = 15
    (model_name, net), = RealMLP_adult.items()
    total_steps = epochs * len(adult_train)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    depths, result_per_batch, result_per_epoch, result_0_and_best_epoch,RealMLP_adult_best = \
        train_and_compute_id(
            epochs,
            RealMLP_adult,
            adult_train,
            adult_test,
            estimator,
            criterion,
            metrics,
            optimizer,
            id_logging_interval,
            device=device,
            batch_scheduler=scheduler,
            depth_fns=depth_fns
        )

    plot_fig5c(model_name, result_0_and_best_epoch,
               depths,50,
               title=f"{model_name}: ID & PC-ID & ED (trained vs untrained)",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9A(result_per_epoch, depths,
               title=f"{model_name}: layer-wise ID across epochs",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9B(result_per_batch,
               title=f"{model_name}: last hidden layer ID & accuracy during training",
               show=show,
               savepath=savepath)
    plot_fig9C(result_per_epoch,
               title=f"{model_name}: test error vs last hidden layer ID",
               show=show,
               savepath=savepath)


    # --- Evaluation of StandardMLP on Adult ---
    (model_name, net), = StandardMLP_adult.items()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    depths, result_per_batch, result_per_epoch, result_0_and_best_epoch, StandardMLP_adult_best = \
        train_and_compute_id(
            epochs,
            StandardMLP_adult,
            adult_train,
            adult_test,
            estimator,
            criterion,
            metrics,
            optimizer,
            id_logging_interval,
            device=device,
            batch_scheduler=scheduler,
            depth_fns=depth_fns
        )

    plot_fig5c(model_name, result_0_and_best_epoch,
               depths,80,
               title=f"{model_name}: ID & PC-ID & ED (trained vs untrained)",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9A(result_per_epoch, depths,
               title=f"{model_name}: layer-wise ID across epochs",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9B(result_per_batch,
               title=f"{model_name}: last hidden layer ID & accuracy during training",
               show=show,
               savepath=savepath)
    plot_fig9C(result_per_epoch,
               title=f"{model_name}: test error vs last hidden layer ID",
               show=show,
               savepath=savepath)


    # --- ID Measurement of Trained Models ---
    id_dynamics_realmlp_trained = compute_id_dynamics_across_models(
        RealMLP_adult_best,
        estimator,
        adult_test,
        depth_fns,
        device,
        pca_dim=False)

    id_dynamics_standardmlp_trained = compute_id_dynamics_across_models(
        StandardMLP_adult_best,
        estimator,
        adult_test,
        depth_fns,
        device,
        pca_dim=False)

    id_dynamics_trained = {**id_dynamics_realmlp_trained,
                        **id_dynamics_standardmlp_trained}

    title = "Intrinsic Dimension on Adult After Training (TwoNN)"
    plot_fig3b(id_dynamics_trained,"TwoNN",title,annotate=True,show=show,savepath=savepath)




def report_id_statistics(model, estimator, train_loader, test_loader, data_name,
                         depth_fns, epochs, optimizer, criterion, metrics, id_logging_interval,
                         device, epoch_scheduler=None, batch_scheduler=None,y_lim=50,
                         show=True,savepath=None):

    (model_name, net), = model.items()
    print(f"--- Starting Analysis for {model_name} on {data_name} ---")

    id_dynamics_untrained = compute_id_dynamics_across_models(
        model,
        estimator,
        test_loader,
        depth_fns,
        device,
        pca_dim=False)
    k, v = id_dynamics_untrained.popitem()
    id_dynamics_untrained[k+'_before'] = v

    depths, result_per_batch, result_per_epoch, result_0_and_best_epoch, model_best = \
        train_and_compute_id(
            epochs,
            model,
            train_loader,
            test_loader,
            estimator,
            criterion,
            metrics,
            optimizer,
            id_logging_interval,
            device=device,
            batch_scheduler=batch_scheduler,
            epoch_scheduler=epoch_scheduler,
            depth_fns=depth_fns
        )

    plot_fig5c(model_name, result_0_and_best_epoch, 
               depths,y_lim,title=f"{model_name}: ID & PC-ID & ED (trained vs untrained)",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9A(result_per_epoch, depths,
               title=f"{model_name}: layer-wise ID across epochs",
               annotate=True,
               show=show,
               savepath=savepath)
    plot_fig9B(result_per_batch,
               title=f"{model_name}: last hidden layer ID & accuracy during training",
               show=show,
               savepath=savepath)
    plot_fig9C(result_per_epoch,
               title=f"{model_name}: test error vs last hidden layer ID",
               show=show,
               savepath=savepath)

    id_dynamics_trained = compute_id_dynamics_across_models(
        model_best,
        estimator,
        test_loader,
        depth_fns,
        device,
        pca_dim=False)

    k, v = id_dynamics_trained.popitem()
    id_dynamics_trained[k+'_after'] = v

    id_dynamics = {**id_dynamics_untrained,
                   **id_dynamics_trained}
    title = (
    f"Intrinsic Dimension on {data_name} "
    f"({model_name}) Before and After Training (TwoNN)"
    )
    plot_fig3b(id_dynamics,"TwoNN",title,annotate=True,show=show,savepath=savepath)




MODELS = {
    "RealMLP": lambda n_num, cat_card, num_classes: RealMLP_TD(n_num, cat_card, num_classes=num_classes),
    "StandardMLP": lambda n_num, cat_card, num_classes: StandardMLP(n_num, cat_card, num_classes=num_classes),
    "TabM": lambda n_num, cat_card, k, num_classes: TabM(n_num, cat_card, k=k, num_classes=num_classes),
}



depth_fns = {"RealMLP":getDepths_RealMLP_TD,
             "StandardMLP":getDepths_StandardMLP,
             "TabM":getDepths_TabM
             }


def evaluate_model_on_data(data_id, model_name, opt_lr, opt_wd,num_classes=None, num_train=20000, num_test=2000,
                           epochs=15, id_logging_interval=15, estimator=None,
                           depth_fns=depth_fns, device=None, criterion=None, metrics=None,
                           y_lim=50,k=10, savepath=None, show=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if estimator is None:
        estimator = {"TwoNN": partial(twonn_id, device=device, batch=512)}

    data = fetch_openml(data_id=data_id, as_frame=True)
    data_name = data.details.get("name", f"openml_{data_id}")
    train_loader, test_loader, num_numerical, cardinality = make_dataloaders(data, num_train, num_test, seed=42)
    if model_name == "RealMLP":
        model = {model_name: MODELS[model_name](num_numerical, cardinality, num_classes)}
        model[model_name].fit_statistics(
            train_loader.dataset.x_num,
            train_loader.dataset.x_cat
        )
    elif model_name == "StandardMLP":
        model = {model_name: MODELS[model_name](num_numerical, cardinality, num_classes)}
        model[model_name].fit_statistics(train_loader.dataset.x_num)
    elif model_name == "TabM":
        model = {model_name: MODELS[model_name](num_numerical, cardinality, k, num_classes)}
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(MODELS.keys())}")

    total_steps = epochs * len(train_loader)
    (model_name, net), = model.items()
    
    if criterion is None:
        if model_name == "TabM":
            if num_classes is None:
                criterion = regress_mse_ensemble_sum
            else:
                criterion = classif_ce_ensemble_sum
        else:
            if num_classes is None:
                criterion = regress_mse_sum
            else:
                criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    if metrics is None:
        if model_name == "TabM":
            if num_classes is None:
                metrics = regress_mse_ensemble_sum
            else:
                metrics = classif_accuracy_ensemble_sum
        else:
            if num_classes is None:
                metrics = regress_mse_sum
            else:
                metrics = accuracy_sum

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=opt_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=opt_wd,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    report_id_statistics(
        model,
        estimator,
        train_loader,
        test_loader,
        data_name,
        depth_fns,
        epochs,
        optimizer,
        criterion,
        metrics,
        id_logging_interval,
        device,
        batch_scheduler=scheduler,
        y_lim=y_lim,
        show=show,
        savepath=savepath
    )




def main():
    p = argparse.ArgumentParser(description="Task 2 runner (tabular)")

    # choose which pipeline to run
    p.add_argument("--adult_full", action="store_true",
                   help="Run the full Adult walkthrough experiment (adult_id_full_experiment).")
    p.add_argument("--eval", action="store_true",
                   help="Run the general evaluation pipeline (evaluate_model_on_data).")

    # common I/O
    p.add_argument("--show", action="store_true", help="Show figures interactively.")
    p.add_argument("--savepath", type=str, default=None,
                   help="Directory to save figures (e.g., figures/task2).")

    # device
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="Force device. Default: auto-detect.")

    # params for eval pipeline
    p.add_argument(
        "--data_id",
        type=int,
        default=1590,
        help="OpenML dataset id (default: 1590=adult). Examples: 1590 (adult), 45551 (higgs), 150 (covertype)."
    )
    p.add_argument("--model", choices=list(MODELS.keys()), default="RealMLP")
    p.add_argument("--num_classes", type=int, default=2,
                   help=(
                       "Number of classes for classification. "
                         "Examples: adult=2, higgs=2, covertype=7. "
                         "Must match the dataset label space."
                         )
    )
    p.add_argument("--opt_lr", type=float, default=1e-2, help="Adam learning rate.")
    p.add_argument("--opt_wd", type=float, default=1e-2, help="Adam weight decay.")
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--id_logging_interval", type=int, default=15)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--y_lim", type=int, default=50)

    args = p.parse_args()

    # default behavior: if nothing chosen, run both on default dataset/model
    if not args.adult_full and not args.eval:
        args.adult_full = True
        args.eval = True

    device = None if args.device is None else torch.device(args.device)

    if args.savepath is not None:
        Path(args.savepath).mkdir(parents=True, exist_ok=True)

    if args.adult_full:
        adult_id_full_experiment(
            show=args.show,
            savepath=args.savepath,
            device=device,
            epochs=args.epochs,
            id_logging_interval=args.id_logging_interval,
        )

    if args.eval:
        evaluate_model_on_data(
            data_id=args.data_id,
            model_name=args.model,
            num_classes=args.num_classes,
            opt_lr=args.opt_lr,
            opt_wd=args.opt_wd,
            num_train=args.num_train,
            num_test=args.num_test,
            epochs=args.epochs,
            id_logging_interval=args.id_logging_interval,
            device=device,
            y_lim=args.y_lim,
            k=args.k,
            show=args.show,
            savepath=args.savepath,
        )

if __name__ == "__main__":
    main()
