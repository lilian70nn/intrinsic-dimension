import os
from functools import partial
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader,Subset
from torchvision.models import alexnet, resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg11, VGG11_Weights, vgg11_bn, VGG11_BN_Weights, vgg13, VGG13_Weights, vgg13_bn, VGG13_BN_Weights
from torchvision.models import vgg16, VGG16_Weights, vgg16_bn, VGG16_BN_Weights, vgg19, VGG19_Weights, vgg19_bn, VGG19_BN_Weights
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, AlexNet_Weights

from src.depths import getDepths, getResNetsDepths, getDepths_cifar_resnet
from src.cnn_models import VGG, AlexNet, ResNet18
from src.id.estimator import twonn_id, mle_id, repeat_compute
from src.postprocess import merged_over_categories
from src.datasets import PathListDataset
from src.id.compute import compute_id_dynamics_across_models
from src.train_and_test import train_and_compute_id
from src.plots import plot_fig3b, plot_fig5c, plot_fig9A, plot_fig9B, plot_fig9C


def task1_1_fig3b_pretrained_cnn_id(show=True,savepath=None,device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Pretrained CNN models, their depth definitions, and intrinsic dimension estimators ---
    models = {
        "VGG19": vgg19(weights=VGG19_Weights.IMAGENET1K_V1),
        "VGG16": vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        "VGG13": vgg13(weights=VGG13_Weights.IMAGENET1K_V1),
        "VGG11": vgg11(weights=VGG11_Weights.IMAGENET1K_V1),
        "VGG19_BN": vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1),
        "VGG16_BN": vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1),
        "VGG13_BN": vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1),
        "VGG11_BN": vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1),
        "AlexNet": alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        "ResNet152": resnet152(weights=ResNet152_Weights.IMAGENET1K_V1),
        "ResNet101": resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
        "ResNet50": resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        "ResNet34": resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
        "ResNet18": resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    }


    depth_fns = {
        "VGG19": getDepths,
        "VGG16": getDepths,
        "VGG13": getDepths,
        "VGG11": getDepths,
        "VGG19_BN": getDepths,
        "VGG16_BN": getDepths,
        "VGG13_BN": getDepths,
        "VGG11_BN": getDepths,
        "AlexNet": getDepths,
        "ResNet152": getResNetsDepths,
        "ResNet101": getResNetsDepths,
        "ResNet50": getResNetsDepths,
        "ResNet34": getResNetsDepths,
        "ResNet18": getResNetsDepths,
    }

    estimators = {"TwoNN":partial(repeat_compute,estimator=partial(twonn_id,device=device,batch=256)),
                "MLE":partial(repeat_compute,estimator=partial(mle_id,device=device,batch=256))
                }


    # --- input preprocessing ---
    data_transform = transforms.Compose([
            transforms.Resize( (224,224) , interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    # --- ImageNet data source and selected categories  ---
    #src_dir = "/content/drive/MyDrive/imagenet_training_single_objs"
    src_dir = "data/imagenet_training_single_objs"
    file_names = ["n01882714/0","n02086240/0","n02087394/0","n02094433/0","n02100583/0","n02100735/0","n02279972/0"]


    ID_all_category = []
    for i,name in enumerate(file_names):

        print(f"Category{i+1}/{len(file_names)}")
        file_name = os.path.join(src_dir,name)
        all_imgs = [os.path.join(file_name, f) for f in os.listdir(file_name) if f.endswith(".JPEG")]
        dataset = PathListDataset(all_imgs, transform=data_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True,num_workers=2)

        id_for_one_category = compute_id_dynamics_across_models(models,
                                                                estimators,
                                                                dataloader,
                                                                depth_fns,
                                                                device,
                                                                pca_dim=False)
        ID_all_category.append(id_for_one_category)


    merged_ID_all_category = merged_over_categories(ID_all_category)

    for estimator in estimators:
        title = f"Intrinsic Dimension on ImageNet ({estimator})"
        plot_fig3b(merged_ID_all_category,estimator,title,show=show,savepath=savepath)
    
    return merged_ID_all_category





def task1_2_fig5c_fig9_training_dynamics(show=True,
                                    savepath=None,
                                    device=None,
                                    epochs=20,
                                    num_samples=500,
                                    id_logging_interval=45,):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # CIFAR-10 data setup for training and intrinsic-dimension experiments.
    # - Applies standard data augmentation to training set (crop + flip).
    # - Creates a small image subset of the test set


    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform_train,
    )

    test_set = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)

    #num_samples = 500
    indices = torch.randperm(len(test_set))[:num_samples]
    small_test_set = Subset(test_set, indices)

    small_test_loader = DataLoader(
        small_test_set,
        batch_size=100,
        shuffle=False,
        num_workers=2,
    )


    #epochs = 20
    #id_logging_interval = 45
    models_to_run = [
        ("vgg16",  VGG("VGG16")),
        ("AlexNet", AlexNet()),
        ("resnet18", ResNet18())
    ]
    depth_fns = {"resnet18": getDepths_cifar_resnet,
                "vgg16":getDepths,
                "AlexNet":getDepths}


    for model_name, net in models_to_run:

        print(f"\n===== Running model: {model_name} =====")

        model = {model_name: net}
        estimator = {"TwoNN":partial(twonn_id,device=device,batch=256)}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
        )
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        depths, result_per_batch, result_per_epoch, result_0_and_best_epoch,_ = \
            train_and_compute_id(
                epochs,
                model,
                train_loader,
                small_test_loader,
                estimator,
                criterion,
                optimizer,
                id_logging_interval,
                device,
                epoch_scheduler=scheduler,
                depth_fns=depth_fns
            )

        plot_fig5c(model_name, result_0_and_best_epoch, 
                   depths,400,
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





def main():
    parser = argparse.ArgumentParser(description="Task 1 experiments")

    parser.add_argument("--task1_1", action="store_true",
                        help="Run Task 1.1 (ImageNet intrinsic dimension)")
    parser.add_argument("--task1_2", action="store_true",
                        help="Run Task 1.2 (CIFAR training ID dynamics)")

    parser.add_argument("--show", action="store_true",
                        help="Show figures instead of only saving")

    parser.add_argument("--savepath", type=str, default=None,
                        help="Directory to save figures")
    
    parser.add_argument("--epochs", type=int, default=20,
                    help="(Task 1.2) Number of training epochs")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="(Task 1.2) Size of the test subset used for ID computation")
    parser.add_argument("--id_logging_interval", type=int, default=45,
                        help="(Task 1.2) Log ID every N mini-batches")

    args = parser.parse_args()

    run_1 = args.task1_1
    run_2 = args.task1_2


    if not run_1 and not run_2:
        run_1 = True
        run_2 = True

    if run_1:
        print(">>> Running Task 1.1")
        task1_1_fig3b_pretrained_cnn_id(
            show=args.show,
            savepath=args.savepath
        )

    if run_2:
        print(">>> Running Task 1.2")
        task1_2_fig5c_fig9_training_dynamics(
        show=args.show,
        savepath=args.savepath,
        epochs=args.epochs,
        num_samples=args.num_samples,
        id_logging_interval=args.id_logging_interval,
    )


if __name__ == "__main__":
    main()