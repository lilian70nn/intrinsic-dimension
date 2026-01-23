import random
from functools import partial
import argparse
import os
import nltk

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import DataCollatorWithPadding,RobertaTokenizer, RobertaModel
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import spearmanr
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample)
from scipy.stats import ttest_ind
import torchvision.models as models


from src.id.estimator import twonn_id,mle_id
from src.depths import getDepths
from src.id.compute import compute_id_dynamics_across_models
from src.plots import title_to_filename





def tokenization(batch, tokenizer, text_key, max_length=512):
    return tokenizer(batch[text_key], truncation=True, padding=True,max_length=max_length)

def compute_id_nlp(X,model,tokenizer,text_key,fraction=0.9,verbose=False,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenize_fn = lambda batch: tokenization(batch, tokenizer, text_key, max_length=512)
    dataset = X.map(tokenize_fn, batched=True,remove_columns=X.column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=10,
                            shuffle=True, collate_fn=collator, num_workers=0)
    hidden_states = []
    for i, data in tqdm(enumerate(dataloader,0)):
        data = {k: v.to(device) for k, v in data.items()}
        with torch.no_grad():
            output = model(**data)
        batch_hidden_states = [h.detach().cpu() for h in output.hidden_states]
        if i == 0:
            hidden_states = batch_hidden_states
        else:
            hidden_states = [torch.cat([h1,h2], dim=0) for h1,h2 in zip(hidden_states,batch_hidden_states)]
    hidden_states_cls = [h[:,0,:].numpy() for h in hidden_states[1:]]


    ids = []
    for h in hidden_states_cls:
        id = twonn_id(torch.tensor(h,dtype=torch.float32),device,batch=256)
        ids.append(id)
    return ids


def task3_1(model=None, tokenizer=None, num_data=250, run=10, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if model is None:
        model = RobertaModel.from_pretrained('roberta-base',output_hidden_states=True).to(device)
    model.eval()
    os.makedirs("figures/task3", exist_ok=True)

    # --- CNN News Dataset ---

    n = num_data
    all_ids_cnn = []
    ds_cnn = load_dataset("cnn_dailymail",'3.0.0',split="train")
    for i in range(run):
        indices = random.sample(range(len(ds_cnn)), n)
        dataset = ds_cnn.select(indices)
        ids = compute_id_nlp(dataset,model,tokenizer,text_key="article")
        all_ids_cnn.append(ids)

    all_ids_cnn = np.array(all_ids_cnn)
    mean_ids_cnn = np.mean(all_ids_cnn, axis=0)
    std_ids_cnn = np.std(all_ids_cnn, axis=0)
    layers_cnn = np.arange(len(mean_ids_cnn))

    plt.figure(figsize=(8,5))

    plt.plot(layers_cnn, mean_ids_cnn, marker="o", label="Mean ID")

    plt.fill_between(layers_cnn, mean_ids_cnn-std_ids_cnn, mean_ids_cnn+std_ids_cnn, alpha=0.2, label="±1 std")

    plt.xlabel("Layer")
    plt.ylabel("ID Score")
    title = "Intrinsic Dimensionality of Cnn News across Layers"
    plt.title(title)
    plt.legend()

    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()



    # --- TweetEval Dataset ---
    configs = ["emoji", "emotion", "hate", "irony", "offensive", "sentiment"]

    datasets_tweet = []
    for cfg in configs:
        ds_tweet = load_dataset("cardiffnlp/tweet_eval", cfg, split="train")
        ds_tweet = ds_tweet.remove_columns([c for c in ds_tweet.column_names if c != "text"])
        datasets_tweet.append(ds_tweet)
    ds_all_tweet = concatenate_datasets(datasets_tweet)

    all_ids_tweet = []
    for i in range(run):
        indices = random.sample(range(len(ds_all_tweet)), n)
        dataset = ds_all_tweet.select(indices)
        ids_tweet = compute_id_nlp(dataset, model, tokenizer, text_key="text")
        all_ids_tweet.append(ids_tweet)

    all_ids_tweet = np.array(all_ids_tweet)
    mean_ids_tweet = np.mean(all_ids_tweet, axis=0)
    std_ids_tweet = np.std(all_ids_tweet, axis=0)
    layers_tweet = np.arange(len(mean_ids_tweet))

    plt.figure(figsize=(8,5))

    plt.plot(layers_tweet, mean_ids_tweet, marker="o", label="Mean ID")

    plt.fill_between(layers_tweet, mean_ids_tweet-std_ids_tweet, mean_ids_tweet+std_ids_tweet, alpha=0.2, label="±1 std")

    plt.xlabel("Layer")
    plt.ylabel("ID Score")
    title = "Intrinsic Dimensionality of Tweets across Layers"
    plt.title(title)
    plt.legend()

    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()




# --- Synthetic Data Generation ---
# --- Method A：Linear Factor Model Sampling ---
def gen(n, d, p, noise_dim=0, sigma=1.0):
    Z = np.random.randn(n, d)
    A = np.random.randn(d, p)
    X = Z @ A
    if noise_dim > 0:
        noise = sigma * np.random.randn(n, noise_dim)
        X = np.concatenate([X, noise], 1)
    return torch.from_numpy(X).float()

# --- Method B：GAN-based Restricted Latent Subspace Sampling ---
def sample_restricted_z(num_samples, d, z_dim=128, truncation=0.4):
    z = torch.zeros(num_samples, z_dim)
    z[:, :d] = torch.from_numpy(truncated_noise_sample(batch_size=num_samples, dim_z=d, truncation=truncation))
    return z

def get_sample(model,z,labels,trunc=0.4,batch_size=64,device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    z = z.to(device)
    labels = labels.to(device)

    outs = []

    with torch.inference_mode():
        for i in range(0, z.size(0), batch_size):
            zi = z[i:i+batch_size]
            yi = labels[i:i+batch_size]
            out = model(zi, yi, trunc)
            outs.append(out)
        out = torch.cat(outs, dim=0)
        out = (out + 1) / 2
    return out.to('cpu')


biggan = BigGAN.from_pretrained('biggan-deep-128')
biggan.eval()

def task3_2_1(num_samples=2000, true_ids=None, device=None):
    
    """
    Task – Simulation with known ID:
    Study how ID estimates change with increasing true intrinsic dimension
    using GAN-based restricted latent subspace sampling.
    """
    os.makedirs("figures/task3", exist_ok=True)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if true_ids is None:
        true_ids = [i for i in range(8,129,2)]
    estimate_ids = []
    for id in tqdm(true_ids,desc="computing estimates change with increasing true ID..."):
        z = sample_restricted_z(num_samples=num_samples, d=id)
        labels = one_hot_from_names(["basenji"] * num_samples, batch_size=num_samples)
        labels = torch.from_numpy(labels)
        out = get_sample(biggan,z,labels,device=device)
        X = out.view(out.size(0), -1)
        estimate_id = twonn_id(X,device)
        estimate_ids.append(estimate_id)
    plt.scatter(true_ids,estimate_ids)
    plt.xlabel("True ID")
    plt.ylabel("Estimated ID")
    title = "Latent Dimension vs. Estimated Intrinsic Dimension (TwoNN)"
    plt.title(title)
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()
    
    spearman_corr, p_s = spearmanr(true_ids, estimate_ids)
    print("Spearman's correlation coefficient: {:.3f}".format(spearman_corr))


def task3_2_2(device=None):

    """
    Task – Simulation with known ID:
    Study how ID estimates change with increasing sample size.
    Comparison between:
        (i) Linear factor model sampling
        (ii) GAN-based restricted latent subspace sampling
    """

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("figures/task3", exist_ok=True)
    # --- ID Estimates vs. Sample Size (Linear Sampling) ---

    sample_sizes = [i for i in range(200,20000,200)]
    estimate_ids_2nn = []
    estimate_ids_mle = []
    for size in sample_sizes:
        X = gen(size,16,128,noise_dim=64)
        estimate_ids_2nn.append(twonn_id(X,device,batch=512))
        estimate_ids_mle.append(mle_id(X,device,batch=512))
    plt.plot(sample_sizes,estimate_ids_2nn, color='blue',label="2NN")
    plt.plot(sample_sizes,estimate_ids_mle,color='red',label="MLE")
    plt.xlabel("Sample size")
    plt.ylabel("Estimated ID")
    title = "ID Estimates vs. Sample Size (Linear Sampling)"
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()

    # --- ID Estimates vs. Sample Size (Restricted GAN Latent Sampling) ---
    sample_sizes = [i for i in range(200,15000,400)]
    estimate_ids_2nn = []
    estimate_ids_mle = []

    for sample_size in tqdm(sample_sizes):
        z = sample_restricted_z(num_samples=sample_size, d=16)
        labels = one_hot_from_names(["basenji"] * sample_size, batch_size=sample_size)
        labels = torch.from_numpy(labels)
        out = get_sample(biggan,z,labels,device=device)
        X = F.adaptive_avg_pool2d(out,32).flatten(1)
        estimate_ids_2nn.append(twonn_id(X,device,batch=512))
        estimate_ids_mle.append(mle_id(X,device,batch=512))

    plt.plot(sample_sizes,estimate_ids_2nn, color='blue',label="2NN")
    plt.plot(sample_sizes,estimate_ids_mle,color='red',label="MLE")
    plt.xlabel("Sample size")
    plt.ylabel("Estimated ID")
    title = "ID Estimates vs. Sample Size (Restricted GAN Latent Sampling)"
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()



def task3_2_3(R=20,d=16,p=5000,noise_dim=2000,sigma=1.0,device=None):

    """
    Task – Effect of data regime:
    Compare ID estimation behavior in p >> n versus n >> p settings
    for both TwoNN and MLE estimators.
    """

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("figures/task3", exist_ok=True)
    # --- 1. Generate Data for Two Regimes ---

    # Regime 1: p >> n (The "High-Dim, Low-Sample" case)
    # n=1000, p=7000. Dimension dominates sample size.

    ids_small_2nn = [twonn_id(gen(1000, d, p, noise_dim, sigma),device,batch=512) for _ in range(R)]
    ids_small_mle = [mle_id(gen(1000, d, p, noise_dim, sigma),device,k=15,batch=512) for _ in range(R)]

    # Regime 2: n >> p (The "Large Sample" case)
    # n=20000, p=7000. Sample size dominates dimension.

    ids_large_2nn = [twonn_id(gen(20000, d, p, noise_dim, sigma),device,batch=512) for _ in range(R)]
    ids_large_mle = [mle_id(gen(20000, d, p, noise_dim, sigma),device,k=15,batch=512) for _ in range(R)]


    # --- 2. Visualization Comparison ---

    data = [ids_small_2nn, ids_large_2nn,ids_small_mle,ids_large_mle]
    labels = ["p >> n, twonn", "n >> p, twonn", "p >> n, mle", "n >> p, mle"]

    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.ylabel("Estimated ID")
    title = "Robustness of ID Estimation across High-Dimensional and Large-Sample Regimes"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()

    # --- 3. Statistical Quantification ---

    ids_small_2nn = np.array(ids_small_2nn)
    ids_large_2nn = np.array(ids_large_2nn)
    print("id measured by twonn:")
    print("p>>n:  mean={:.3f}, std={:.3f}".format(ids_small_2nn.mean(), ids_small_2nn.std(ddof=1)))
    print("n>>p:  mean={:.3f}, std={:.3f}".format(ids_large_2nn.mean(), ids_large_2nn.std(ddof=1)))

    t, pval = ttest_ind(ids_small_2nn, ids_large_2nn, equal_var=False)
    print("t-test: t={:.2f}, p={:.2e}".format(t, pval))

    pooled_sd = np.sqrt(((ids_small_2nn.var(ddof=1) + ids_large_2nn.var(ddof=1)) / 2))
    cohen_d = (ids_large_2nn.mean() - ids_small_2nn.mean()) / pooled_sd
    print("Cohen's d = {:.2f}".format(cohen_d))

    print("*"*25)
    ids_small_mle = np.array(ids_small_mle)
    ids_large_mle = np.array(ids_large_mle)
    print("id measured by mle:")
    print("p>>n:  mean={:.3f}, std={:.3f}".format(ids_small_mle.mean(), ids_small_mle.std(ddof=1)))
    print("n>>p:  mean={:.3f}, std={:.3f}".format(ids_large_mle.mean(), ids_large_mle.std(ddof=1)))

    t, pval = ttest_ind(ids_small_mle, ids_large_mle, equal_var=False)
    print("t-test: t={:.2f}, p={:.2e}".format(t, pval))

    pooled_sd = np.sqrt(((ids_small_mle.var(ddof=1) + ids_large_mle.var(ddof=1)) / 2))
    cohen_d = (ids_large_mle.mean() - ids_small_mle.mean()) / pooled_sd
    print("Cohen's d = {:.2f}".format(cohen_d))


def task3_2_4(data_size=3000, d=16, R=8,device=None):

    nltk.download('wordnet')
    nltk.download('omw-1.4')

    """
    Task – Layer-wise analysis:
    Investigate whether early/middle/late layers of CNNs
    recover the true intrinsic dimension of input data.
    """

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs("figures/task3", exist_ok=True)

    # --- 1. Data Generation & Preprocessing ---
    z = sample_restricted_z(num_samples=data_size, d=d)
    all_classes = ["basenji", "goldfish", "soccer ball", "tench", "tabby cat"]
    chosen = random.choices(all_classes, k=data_size)
    labels = one_hot_from_names(chosen, batch_size=data_size)
    labels = torch.from_numpy(labels)
    out = get_sample(biggan,z,labels,device=device)
    out = torch.nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    out = (out - torch.tensor(imagenet_mean)[None, :, None, None]) / torch.tensor(imagenet_std)[None, :, None, None]
    labels = labels.argmax(dim=1)
    dataset = TensorDataset(out, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)


    # --- 2. Model Setup (VGG16) ---
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16.eval()
    vgg16.to(device)
    model = {"vgg16":vgg16}
    estimators = {"TwoNN": partial(twonn_id,device=device,batch=512,subsample=0.85),
                "MLE": partial(mle_id,device=device,batch=512,subsample=0.85)}
    depth_fns = {
        "vgg16":getDepths
    }

    # --- 3. Layer-wise ID Extraction Loop ---
    id_2nn = []
    id_mle = []
    for _ in range(R):
        id_temp = compute_id_dynamics_across_models(model,estimators,loader,depth_fns,device)
        id_2nn.append(id_temp["vgg16"]["TwoNN"])
        id_mle.append(id_temp["vgg16"]["MLE"])
    depths = id_temp["vgg16"]["depths"]



    # --- 4. Visualization ---
    rel = depths/depths[-1]

    plt.figure()
    plt.plot(rel,np.array(id_2nn).mean(axis=0), color='C0', marker='o', label="2NN")
    plt.plot(rel,np.array(id_mle).mean(axis=0), color='C3', marker='o', label="MLE")
    plt.xlabel("Layer depth")
    plt.ylabel("Estimated ID")
    title = "ID estimates across layers"
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()


    plt.figure()
    bp_2nn = plt.boxplot(
        np.array(id_2nn),
        positions=rel,
        widths=0.04,
        patch_artist=False,
        manage_ticks=False
    )
    plt.plot(rel, np.array(id_2nn).mean(axis=0), color="C0", linewidth=2, label="2NN mean")
    plt.xlabel("Layer depth (relative)")
    plt.ylabel("Estimated ID")
    title = "2NN estimates ID across layers"
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()


    plt.figure()
    bp_mle = plt.boxplot(
        np.asarray(id_mle),
        positions=rel,
        widths=0.04,
        patch_artist=False,
        manage_ticks=False
    )

    plt.plot(rel, np.array(id_mle).mean(axis=0), color="C3", linewidth=2, label="MLE mean")
    plt.xlabel("Layer depth (relative)")
    plt.ylabel("Estimated ID")
    title = "MLE estimates ID across layers"
    plt.title(title)
    plt.savefig(f"figures/task3/{title_to_filename(title)}",
            dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "3_1", "3_2_1", "3_2_2", "3_2_3", "3_2_4"]
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task in ["all", "3_1"]:
        task3_1(device=device)
    if args.task in ["all", "3_2_1"]:
        task3_2_1(device=device)
    if args.task in ["all", "3_2_2"]:
        task3_2_2(device=device)
    if args.task in ["all", "3_2_3"]:
        task3_2_3(device=device)
    if args.task in ["all", "3_2_4"]:
        task3_2_4(device=device)