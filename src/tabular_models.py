import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tabm


class Capture(nn.Module):
    # Identity layer used as a hook point for representation analysis
    def forward(self, x):
        return x
    

class StandardMLP(nn.Module):

    """
    StandardMLP: a fully-connected baseline for tabular data.

    This model follows a conventional tabular MLP pipeline:

    Preprocessing:
        - Numerical features are normalized using Z-score normalization
          (mean and standard deviation estimated from training data).
        - Categorical features are encoded using one-hot encoding.
        - Numerical and categorical features are concatenated into a
          single flat feature vector.

    Main Backbone Network Architecture:
        [Linear -> ReLU -> Dropout] x 3 -> Linear -> Logits

    Instrumentation:
        - Explicit identity layers ("Capture") are inserted to expose
          intermediate representations for analysis (e.g., intrinsic
          dimension as a function of depth).

    """


    def __init__(self, num_numerical, cat_cardinalities,
                 dropout=0.5, hidden_dim=256, num_classes=None):


        '''
        Args:
            num_numerical (int): The number of numerical features.
            cat_cardinalities (list[int]): Cardinality for all categorical features.
            dropout (float): Dropout rate applied after ReLU activations.
            hidden_dim (int): Dimensionality of the hidden layers.
            num_classes (int): The number of output classes.
        '''

        super().__init__()

        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities

        total_input_dim = num_numerical + sum(cat_cardinalities)

        # Z-Score Statistics (Mean and Std Dev) are computed in fit_statistics
        self.register_buffer('mean', torch.zeros(1, num_numerical))
        self.register_buffer('std', torch.ones(1, num_numerical))
        self.fitted = False

        # ----- capture checkpoints -----
        self.capture_input = Capture()
        self.capture_after_preprocess = Capture()

        out_dim = 1 if num_classes is None else num_classes

        # --- MLP Core Structure ---
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim)
        )

    def fit_statistics(self, x_num):

        """Computes and stores the mean and standard deviation of numerical
        features for Z-Score normalization."""
        with torch.no_grad():
            mean = x_num.mean(dim=0, keepdim=True)
            std = x_num.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            self.mean.copy_(mean)
            self.std.copy_(std)
            self.fitted = True
        print("StandardMLP Statistics Fitted!")

    def _encode_features(self, x_num, x_cat=None):

        """Performs Z-Score normalization on numerical features and One-Hot Encoding
        on categorical features."""
        x_num_norm = (x_num - self.mean) / self.std
        parts = [x_num_norm]

        if x_cat is not None:
            for i, card in enumerate(self.cat_cardinalities):
                col = x_cat[:, i]
                mask = col >= 0
                safe = col.clamp(min=0)
                oh = F.one_hot(safe, num_classes=card).float()
                oh[~mask] = 0.0
                parts.append(oh)

        return torch.cat(parts, dim=1)


    def forward(self, inputs):

        '''Forward pass through the model.'''
        x_num, x_cat = inputs

        if not self.fitted:
            raise RuntimeError("Call fit_statistics(x_num) before forward().")

        if len(self.cat_cardinalities) == 0 or x_cat.shape[1] == 0:
            x_cat = None

        _ = self.capture_input(x_num if x_cat is None else torch.cat([x_num, x_cat.float()], dim=1))

        x = self._encode_features(x_num, x_cat)
        _ = self.capture_after_preprocess(x)

        return self.mlp(x)
    
    


class LearnableScaling(nn.Module):
    """
    Learnable element-wise feature scaling.

    Applies a trainable multiplicative scale to each feature dimension.
    Used to reweight features after embedding and concatenation.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class PBLDNumEmbed(nn.Module):

    """
    For each numerical feature x_i, this module produces a 4-dimensional embedding
    Numerical embedding (PBLD-style), per feature x_i:
    phi_i(x_i) = cos(2π * w_i * x_i + b_i)         ∈ R^{K}
    g_i(x_i)   = W2_i * phi_i(x_i) + b2_i          ∈ R^{3}
    e_i(x_i)   = [x_i, g_i(x_i)]                   ∈ R^{4}
    All e_i are concatenated to shape (B, n_features*4).
    """

    def __init__(self, n_features, n_freq=16, sigma=0.01):
        super().__init__()
        self.n_features = n_features
        self.n_freq = n_freq

        # per-feature frequencies + phase
        self.w = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        self.b = nn.Parameter(torch.zeros(n_features, n_freq))

        # per-feature projection to 3 dims
        self.W2 = nn.Parameter(torch.randn(n_features, 3, n_freq) * sigma)
        self.b2 = nn.Parameter(torch.zeros(n_features, 3))


    @property
    def out_dim(self):
        return self.n_features * 4

    def forward(self, x):
        # x: (B, F)
        x = x.float()
        ang = 2 * math.pi * x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        phi = torch.cos(ang)   # (B, F, K)

        # linear projection per feature
        proj = torch.einsum("bfk,fok->bfo", phi, self.W2) + self.b2  # (B,F,3)

        out = torch.cat([x.unsqueeze(-1), proj], dim=-1)  # (B,F,4)
        return out.reshape(out.size(0), -1)


class NTLinear(nn.Module):


    """
    Neural Tangent Parameterization (NTP) linear layer.

    Forward:
        y = (1 / sqrt(d_in)) * (W x) + b

    Note on initialization:
        The RealMLP-TD paper uses a data-dependent initialization for weights (row-wise rescaling
        based on data statistics) and for biases (he+5 / hull+5).
        Here we keep the NTP forward parameterization, but use standard PyTorch initialization
        (Kaiming-uniform weights and uniform biases) for simplicity.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = F.linear(x, self.weight, None) / math.sqrt(self.in_features)
        if self.bias is not None:
            y = y + self.bias
        return y

class ParametricActivation(nn.Module):


    def __init__(self, dim, act="selu", init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), init_alpha))

        if act == "selu":
            self.act = nn.SELU()
        elif act == "mish":
            self.act = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x):
        return (1.0 - self.alpha) * x + self.alpha * self.act(x)




class RealMLP_TD(nn.Module):
    """
    RealMLP-TD: a tabular MLP with structured preprocessing and learnable embeddings.

    Pipeline overview:

    1. Input partitioning
        - Numerical features are processed directly.
        - Categorical features are split into:
            * low-cardinality categories → one-hot encoding
            * high-cardinality categories → embedding layers

    2. Robust preprocessing (numerical + low-card OHE only)
        - Feature-wise median centering
        - Robust scaling using inverse IQR
        - Smooth clipping via x / sqrt(1 + (x / a)^2)

    3. Feature embedding
        - Numerical features are mapped via PBLD-style numerical embeddings
        - High-cardinality categorical features use trainable embeddings
        - All features are concatenated into a single representation

    4. Learnable feature scaling
        - Element-wise trainable rescaling of the full representation

    5. MLP backbone
        - [Linear → ParametricActivation → Dropout] × 3 → Linear → logits

    Instrumentation:
        - Identity-based Capture modules are inserted at key semantic stages
          to expose intermediate representations for analysis (e.g. intrinsic dimension).
    """

    def __init__(
        self,
        num_numerical,
        cat_cardinalities=None,
        max_card=8,
        emb_dim=8,
        dropout=0.15,
        hidden_dim=256,
        num_classes=None,
    ):
        super().__init__()

        self.num_numerical = num_numerical
        self.cat_cardinalities = list(cat_cardinalities) if cat_cardinalities else []
        self.max_card = max_card
        self.emb_dim = emb_dim

        # ----- capture checkpoints -----
        self.capture_input = Capture()
        self.capture_after_ohe = Capture()
        self.capture_after_robust = Capture()
        self.capture_after_embed = Capture()
        self.capture_after_scale = Capture()

        # ----- categorical modules -----
        self.cat_processors = nn.ModuleList()
        self.is_embedding = []

        low_cat_dim = 0
        high_cat_dim = 0
        for card in self.cat_cardinalities:
            if card <= max_card:
                self.cat_processors.append(nn.Identity())
                self.is_embedding.append(False)
                if card == 2:
                    low_cat_dim += 1
                else:
                    low_cat_dim += card
            else:
                emb = nn.Embedding(card+1, emb_dim, padding_idx=0)
                nn.init.normal_(emb.weight, std=0.01)
                with torch.no_grad():
                    emb.weight[0].zero_()
                self.cat_processors.append(emb)
                self.is_embedding.append(True)
                high_cat_dim += emb_dim

        # ----- robust stats only for numerical + low-card one-hot -----
        self.robust_dim = num_numerical + low_cat_dim
        self.register_buffer("median", torch.zeros(1, self.robust_dim))
        self.register_buffer("scale", torch.ones(1, self.robust_dim))
        self.fitted = False

        # ----- numerical embedding -----

        self.num_embedding = PBLDNumEmbed(
            n_features=num_numerical,
        )

        # final input dim after embeddings
        total_input_dim = num_numerical * 4 + low_cat_dim + high_cat_dim

        # ----- learnable scaling -----
        self.learnable_scaling = LearnableScaling(total_input_dim)

        if num_classes is None:
            # regression
            act = "mish"
        else:
            # classification
            act = "selu"

        out_dim = 1 if num_classes is None else num_classes
        # ----- MLP -----
        self.mlp = nn.Sequential(
            NTLinear(total_input_dim, hidden_dim),
            ParametricActivation(hidden_dim, act),
            nn.Dropout(dropout),

            NTLinear(hidden_dim, hidden_dim),
            ParametricActivation(hidden_dim, act),
            nn.Dropout(dropout),

            NTLinear(hidden_dim, hidden_dim),
            ParametricActivation(hidden_dim, act),
            nn.Dropout(dropout),

            NTLinear(hidden_dim, out_dim),
        )

    # ---------- encoders ----------
    def _encode_low(self, x_num, x_cat=None):
        outs = [x_num]
        if x_cat is None or len(self.cat_cardinalities) == 0:
            return torch.cat(outs, dim=1)
        for i, proc in enumerate(self.cat_processors):
            K = self.cat_cardinalities[i]
            if self.is_embedding[i]:
                continue

            col = x_cat[:, i]
            if K == 2:
                # binary -> 1 dim in {-1, +1}, missing -> 0
                # col==0 -> -1, col==1 -> +1, col==-1 -> 0
                out = torch.where(col < 0, torch.zeros_like(col), col * 2 - 1).float().unsqueeze(1)
                outs.append(out)
            else:
                mask = col >= 0
                safe = col.clamp(min=0) 
                oh = F.one_hot(safe, num_classes=K).float()
                oh[~mask] = 0.0
                outs.append(oh)
        return torch.cat(outs, dim=1)
    

    # High-card categorical encoding convention:
    # input indices: -1 for missing, 0..(card-1) for observed categories
    # embedding uses padding_idx=0 for missing => shift observed categories by +1
    def _encode_high(self, x_cat):
        if x_cat is None or len(self.cat_cardinalities) == 0:
            return None

        outs = []
        for i, proc in enumerate(self.cat_processors):
            if self.is_embedding[i]:
                col = x_cat[:, i]
                col = torch.where(col < 0, torch.zeros_like(col), col + 1) 
                outs.append(proc(col))
        if len(outs) == 0:
            return None
        return torch.cat(outs, dim=1)

    # ---------- stats ----------
    def fit_statistics(self, x_num, x_cat=None):
        self.eval()
        with torch.no_grad():
            X = self._encode_low(x_num, x_cat)
            median = torch.quantile(X, 0.5, dim=0, keepdim=True)

            q25 = torch.quantile(X, 0.25, dim=0)
            q75 = torch.quantile(X, 0.75, dim=0)
            iqr = q75 - q25

            q0 = torch.quantile(X, 0.0, dim=0)
            q1 = torch.quantile(X, 1.0, dim=0)
            rng = q1 - q0
            
            scale = torch.zeros_like(iqr)
            mask_iqr = iqr != 0
            scale[mask_iqr] = 1.0 / iqr[mask_iqr]

            mask_rng = (~mask_iqr) & (rng != 0)
            scale[mask_rng] = 2.0 / rng[mask_rng]

            self.median.copy_(median)
            self.scale.copy_(scale.unsqueeze(0))
            self.fitted = True

        self.train()
        print("RealMLP statistics fitted.")

    def _forward_to_mlp_input(self, x_num, x_cat=None):

        if not self.fitted:
            raise RuntimeError("Call fit_statistics(x_num, x_cat) before forward().")

        # collect raw high-card indices for later capture points
        raw_high_cat = None
        if x_cat is not None:
            cols = []
            for i, proc in enumerate(self.cat_processors):
                if self.is_embedding[i]:
                    col = x_cat[:, i]
                    cols.append(col)
            if len(cols) > 0:
                raw_high_cat = torch.stack(cols, dim=1).float()

        # capture: raw input
        _ = self.capture_input(x_num if x_cat is None else torch.cat([x_num, x_cat.float()], dim=1))

        # robust preprocessing on numerical + low-card OHE
        x_low = self._encode_low(x_num, x_cat)
        # capture: after one-hot
        _ = self.capture_after_ohe(x_low if raw_high_cat is None else torch.cat([x_low, raw_high_cat], dim=1))

        # robust preprocessing (median / IQR / smooth clip)
        x_low = (x_low - self.median) * self.scale
        a = 3.0
        x_low = x_low / torch.sqrt(1 + (x_low / a) ** 2)
        # capture: after robust preprocessing
        _ = self.capture_after_robust(x_low if raw_high_cat is None else torch.cat([x_low, raw_high_cat], dim=1))


        x_num_proc = x_low[:, : self.num_numerical]
        x_low_ohe = x_low[:, self.num_numerical :]

        # numerical embedding (PBLD)
        x_num_emb = self.num_embedding(x_num_proc)
        # high-card categorical embedding
        x_high = self._encode_high(x_cat)

        if x_high is None:
            x = torch.cat([x_num_emb, x_low_ohe], dim=1)
        else:
            x = torch.cat([x_num_emb, x_low_ohe, x_high], dim=1)
        # capture: after embedding
        _ = self.capture_after_embed(x)

        # learnable scaling
        x = self.learnable_scaling(x)
        # capture: after scaling
        _ = self.capture_after_scale(x)

        return x



    def forward(self, inputs):

        x_num, x_cat = inputs

        if len(self.cat_cardinalities) == 0 or x_cat.shape[1] == 0:
            x_cat = None

        x = self._forward_to_mlp_input(x_num, x_cat)
        return self.mlp(x)


class TabM(nn.Module):

    """
    Initializes the TabM model, which uses the BatchEnsemble technique to run k 
    ensemble members simultaneously via shared weights and per-member (R,S,B) adapters..

    Preprocessing Pipeline:
        1. Feature Encoding: Numerical features are used directly; Categorical features are mapped via Embeddings.
        2. Concatenation: Joins numerical features with all categorical Embeddings to form the backbone input.
    Main Backbone Network:
        Architecture: 
        [LinearBatchEnsemble -> ReLU -> Dropout] x n_blocks -> LinearEnsemble -> Logits((B, k, C))

        Architecture Note:
            The EnsembleView layer expands the input to (Batch, k, D), In the first BatchEnsemble layer, 
            only the input scaling R is initialized with random signs; S is set to ones. All subsequent layers 
            initialize R and S to ones, bias B is zero-initialized.

            Capture (identity) modules are inserted as hook anchors for representation analysis.
    """

    def __init__(self, num_numerical, cat_cardinalities,
                 d_block=512, num_classes=None, n_blocks=3, k=16, dropout=0.1, emb_dim=8):


        """
        Args:
            k (int): The number of ensemble members (Ensemble Size). 
                    Paper uses k=32; we use a smaller k for ID measurements
            d_block (int): Dimensionality of each MLP block.
            n_blocks (int): The number of MLP blocks.
            num_classes (int): The number of output classes.
            (Other standard args omitted for brevity)
        """

        super().__init__()

        # ----- capture checkpoints -----
        self.capture_input = Capture()
        self.capture_after_preprocess = Capture()
        self.cat_cardinalities = cat_cardinalities
        

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c+1, emb_dim, padding_idx=0) for c in self.cat_cardinalities
        ])

        for emb in self.cat_embeddings:
            with torch.no_grad():
                emb.weight[0].zero_()

        backbone_input_dim = num_numerical + len(self.cat_cardinalities) * emb_dim

        out_dim = 1 if num_classes is None else num_classes

        layers = []
        # --- 1. Ensemble View Layer (Core) ---
        layers.append(tabm.EnsembleView(k=k))
        # --- 2. TabM Blocks ---
        # initialize all multiplicative adapters R and S, except for the very first R, deterministically with 1
        for i in range(n_blocks):
            current_in = backbone_input_dim if i == 0 else d_block
            scaling_init = ('random-signs', 'ones') if i == 0 else ('ones', 'ones')

            be_layer = tabm.LinearBatchEnsemble(
                in_features=current_in,
                out_features=d_block,
                k=k,
                bias=True,
                scaling_init=scaling_init,
            )
            # B = 0
            if be_layer.bias is not None:
                with torch.no_grad():
                    be_layer.bias.zero_()

            layers.extend([
                be_layer,
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        # --- 3. Final Output Layer ---
        layers.append(tabm.LinearEnsemble(in_features=d_block, out_features=out_dim, k=k))

        self.ensemble_mlp = nn.Sequential(*layers)

    def forward(self, inputs):

        x_num, x_cat = inputs

        if len(self.cat_cardinalities) == 0 or x_cat.shape[1] == 0:
            x_cat = None

        _ = self.capture_input(x_num if x_cat is None else torch.cat([x_num, x_cat.float()], dim=1))

        emb_outs = []
        if x_cat is not None:
            for i, emb in enumerate(self.cat_embeddings):
                col = x_cat[:, i]
                col = torch.where(col < 0, torch.zeros_like(col), col + 1)
                emb_outs.append(emb(col))
        
        x = torch.cat([x_num] + emb_outs, dim=1)
        _ = self.capture_after_preprocess(x)

        return self.ensemble_mlp(x)