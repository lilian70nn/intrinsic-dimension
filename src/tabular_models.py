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
                 dropout=0.5, hidden_dim=256, num_classes=2):


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

            nn.Linear(hidden_dim, num_classes)
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
                oh = F.one_hot(col, num_classes=card).float()
                parts.append(oh)

        return torch.cat(parts, dim=1)

    def forward(self, x_num, x_cat=None):

        '''Forward pass through the model.'''

        if isinstance(x_num, (tuple, list)):
            x_cat = x_num[1]
            x_num = x_num[0]

        if not self.fitted:
            raise RuntimeError("Call fit_statistics(x_num) before forward().")

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
    Minimal PBLD-style numerical embedding:
      phi(x) = [cos(2π w x + b), sin(2π w x + b)]
      h1 = Linear(phi)
      h2 = Linear([phi, h1])   # Dense concat
      out = Linear([phi, h1, h2]) -> d_embedding
    Returns (B, n_features * d_embedding)
    """
    def __init__(self, n_features, d_embedding=8, n_freq=16, sigma=0.01):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.n_freq = n_freq

        # per-feature frequencies + phase (cos-bias)
        self.w = nn.Parameter(torch.randn(n_features, n_freq) * sigma)
        self.b = nn.Parameter(torch.zeros(n_features, n_freq))

        phi_dim = 2 * n_freq
        self.lin1 = nn.Linear(phi_dim, phi_dim, bias=True)
        self.lin2 = nn.Linear(phi_dim + phi_dim, phi_dim, bias=True)
        self.proj = nn.Linear(phi_dim + phi_dim + phi_dim, d_embedding, bias=True)

    @property
    def out_dim(self):
        return self.n_features * self.d_embedding

    def forward(self, x):
        # x: (B, n_features)
        x = x.float()
        ang = 2 * math.pi * x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)  # (B,F,K)
        phi = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)  # (B,F,2K)

        B, F, D = phi.shape
        phi2 = phi.reshape(B * F, D)

        h1 = self.lin1(phi2)
        h2 = self.lin2(torch.cat([phi2, h1], dim=-1))
        out = self.proj(torch.cat([phi2, h1, h2], dim=-1))  # (B*F, d_emb)

        return out.reshape(B, F * self.d_embedding)

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
        - [Linear → PReLU → Dropout] × 3 → Linear → logits

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
        num_emb_dim=8,
        dropout=0.15,
        hidden_dim=256,
        num_classes=2,
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
                low_cat_dim += card
            else:
                emb = nn.Embedding(card, emb_dim)
                nn.init.normal_(emb.weight, std=0.01)
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
            d_embedding=num_emb_dim,
        )

        # final input dim after embeddings
        total_input_dim = num_numerical * num_emb_dim + low_cat_dim + high_cat_dim

        # ----- learnable scaling -----
        self.learnable_scaling = LearnableScaling(total_input_dim)

        # ----- MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    # ---------- encoders ----------
    def _encode_low(self, x_num, x_cat):
        outs = [x_num]
        if x_cat is None or len(self.cat_cardinalities) == 0:
            return torch.cat(outs, dim=1)
        for i, proc in enumerate(self.cat_processors):
            if not self.is_embedding[i]:
                col = x_cat[:, i]
                outs.append(F.one_hot(col, num_classes=self.cat_cardinalities[i]).float())
        return torch.cat(outs, dim=1)

    def _encode_high(self, x_cat):
        if x_cat is None or len(self.cat_cardinalities) == 0:
            return None

        outs = []
        for i, proc in enumerate(self.cat_processors):
            if self.is_embedding[i]:
                col = x_cat[:, i]
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

            q1 = torch.quantile(X, 0.25, dim=0)
            q3 = torch.quantile(X, 0.75, dim=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1.0

            scale = 1.0 / iqr

            self.median.copy_(median)
            self.scale.copy_(scale.unsqueeze(0))
            self.fitted = True

        self.train()
        print("RealMLP statistics fitted.")

    def forward(self, x_num, x_cat=None):
        if not self.fitted:
            raise RuntimeError("Call fit_statistics(x_num, x_cat) before forward().")

        if isinstance(x_num, (tuple, list)):
            x_cat = x_num[1]
            x_num = x_num[0]

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

        return self.mlp(x)


class TabM(nn.Module):

    """
    Initializes the TabM model, which uses the BatchEnsemble technique to run 'k' independent
    sub-networks simultaneously using parameter sharing.

    Preprocessing Pipeline:
        1. Feature Encoding: Numerical features are used directly; Categorical features are mapped via Embeddings.
        2. Concatenation: Joins numerical features with all categorical Embeddings to form the backbone input.
    Main Backbone Network:
        Architecture: [LinearBatchEnsemble -> ReLU -> Dropout] x n_blocks -> LinearEnsemble -> Logits

        Architecture Note:
            The EnsembleView layer expands the input to (Batch, k, D), the first BatchEnsemble block uses
            random-sign scaling initialization, while subsequent blocks are initialized as identity (ones),
            following the TabM initialization scheme.

            Capture (identity) modules are inserted as hook anchors for representation analysis.
    """

    def __init__(self, num_numerical, cat_cardinalities,
                 d_block=512, num_classes=2, n_blocks=3, k=16, dropout=0.1, emb_dim=8):


        """
        Args:
            k (int): The number of ensemble members (Ensemble Size).
            d_block (int): Dimensionality of each MLP block.
            n_blocks (int): The number of MLP blocks.
            num_classes (int): The number of output classes.
            (Other standard args omitted for brevity)
        """

        super().__init__()

        # ----- capture checkpoints -----
        self.capture_input = Capture()
        self.capture_after_preprocess = Capture()

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c, emb_dim) for c in cat_cardinalities
        ])

        backbone_input_dim = num_numerical + len(cat_cardinalities) * emb_dim


        layers = []
        # --- 1. Ensemble View Layer (Core) ---
        layers.append(tabm.EnsembleView(k=k))
        # --- 2. TabM Blocks ---
        # initialize all multiplicative adapters R and S, except for the very first one, deterministically with 1
        for i in range(n_blocks):
            current_in = backbone_input_dim if i == 0 else d_block
            scaling_init = 'random-signs' if i == 0 else 'ones'

            layers.extend([
                tabm.LinearBatchEnsemble(
                    in_features=current_in,
                    out_features=d_block,
                    k=k,
                    scaling_init=scaling_init
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        # --- 3. Final Output Layer ---
        layers.append(tabm.LinearEnsemble(in_features=d_block, out_features=num_classes, k=k))

        self.ensemble_mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat=None):

        if isinstance(x_num, (tuple, list)):
            x_cat = x_num[1]
            x_num = x_num[0]

        _ = self.capture_input(x_num if x_cat is None else torch.cat([x_num, x_cat.float()], dim=1))

        emb_outs = []
        if x_cat is not None:
            emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat([x_num] + emb_outs, dim=1)
        _ = self.capture_after_preprocess(x)

        return self.ensemble_mlp(x)