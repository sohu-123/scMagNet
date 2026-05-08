import os
import warnings
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import softmax, cosine_similarity
import logging
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import torch.nn.functional as F
warnings.filterwarnings("ignore")


def setup_device():
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def create_similarity_graph(Y, k=15):
    n_spots = Y.shape[0]
    k = min(k, n_spots - 1)
    knn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    distances, indices = knn.kneighbors(Y)
    S = np.zeros((n_spots, n_spots))
    for i in range(n_spots):
        sigma = max(np.median(distances[i, 1:]), 1e-8)
        for j_idx, j in enumerate(indices[i]):
            if i != j:
                S[i, j] = np.exp(-distances[i, j_idx] ** 2 / (2 * sigma ** 2))
    if np.sum(S) == 0:
        S += np.random.random(S.shape) * 0.01
    return torch.FloatTensor(S)


def find_highly_variable_genes(adata, n_top_genes=1000):
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    except:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    return np.where(adata.var['highly_variable'])[0]


def find_common_genes_from_hvg(adata_sc, adata_ST, n_top_genes=1000):
    adata_sc1 = adata_sc.copy()
    adata_ST1 = adata_ST.copy()
    sc_genes = set(adata_sc.var_names[find_highly_variable_genes(adata_sc1, n_top_genes)])
    st_genes = set(adata_ST.var_names[find_highly_variable_genes(adata_ST1, n_top_genes)])
    return sorted(sc_genes & st_genes)


def compute_scale_factor(X_sc, X_st):
    return torch.sum(X_st) / torch.sum(X_sc)


def initialize_assignment_matrix(model, X_sc, X_st, n_neighbors=15):
    device = setup_device()
    X_sc_np = X_sc.cpu().numpy() if torch.is_tensor(X_sc) else X_sc
    X_st_np = X_st.cpu().numpy() if torch.is_tensor(X_st) else X_st
    combined = np.vstack([X_sc_np, X_st_np])
    combined = np.nan_to_num(combined, nan=0.0)
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalized = combined / row_sums * row_sums.mean()
    pca = PCA(n_components=min(50, normalized.shape[1]))
    sc_pca = pca.fit_transform(normalized)[:X_sc_np.shape[0]]
    st_pca = pca.transform(normalized)[X_sc_np.shape[0]:]
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, sc_pca.shape[0] - 1)).fit(sc_pca)
    _, indices = knn.kneighbors(st_pca)
    A_init = torch.zeros(model.n_st_spots, model.n_sc_cells)
    A_init = np.log(1e-10) * A_init
    for i in range(model.n_st_spots):
        A_init[i, indices[i]] = np.log(1.0 / len(indices[i]))
    A_init = torch.tensor(np.random.normal(0, 1, (model.n_st_spots, model.n_sc_cells)), dtype=torch.float32)
    model.A.data = A_init.to(device)
    return sc_pca


def load_and_align_data(sc_path, st_path, n_top_genes=1000, common_genes=None, fix_genes=False):
    adata_ST = sc.read(st_path)
    adata_sc = sc.read(sc_path)
    sc.pp.normalize_total(adata_sc)
    if common_genes is None:
        common_genes = find_common_genes_from_hvg(adata_sc, adata_ST, n_top_genes)
    print('the number of common gene are', len(common_genes))
    assert len(common_genes) >= 100, "Too few common HVGs (<100)"
    if not fix_genes:
        common_genes = list(set(common_genes) & set(adata_ST.var_names))
        common_genes = list(set(common_genes) & set(adata_sc.var_names))
    X_sc = adata_sc[:, common_genes].X.toarray() if hasattr(adata_sc.X, 'toarray') else adata_sc[:, common_genes].X
    X_st = adata_ST[:, common_genes].X.toarray() if hasattr(adata_ST.X, 'toarray') else adata_ST[:, common_genes].X
    X_sc, X_st = torch.FloatTensor(X_sc), torch.FloatTensor(X_st)
    Y = torch.FloatTensor(adata_ST.obsm['spatial']) if 'spatial' in adata_ST.obsm else None
    return X_sc, X_st, Y, adata_sc, adata_ST, common_genes


def transform_quantile_clip(X: torch.Tensor, q: float = 0.995) -> torch.Tensor:
    """先按基因 clip 到 q 分位，再 log1p，压制极端高表达基因。"""
    thresholds = torch.quantile(X, q, dim=0, keepdim=True)
    X_clipped  = torch.clamp(X, max=thresholds)
    return torch.log1p(X_clipped)


class VAEImprovedSpatialModel(nn.Module):
    def __init__(self, n_sc_cells, n_st_spots, n_genes, embedding_dim=8,
                 lambda_reg=0, lambda_l2=1, lambda_l4=0, lambda_l5=0,
                 lambda_ot=-1, lambda_M=1, lambda_M1=1, lambda_r=0,
                 lambda_kl=0.1, lambda_pca=1, lambda_f=0, lambda_row_sum=1.0,
                 predictA=None):
        super().__init__()
        self.n_sc_cells    = n_sc_cells
        self.n_st_spots    = n_st_spots
        self.n_genes       = n_genes
        self.embedding_dim = embedding_dim
        self.lambda_reg    = lambda_reg
        self.lambda_l2     = lambda_l2
        self.lambda_l4     = lambda_l4
        self.lambda_l5     = lambda_l5
        self.lambda_ot     = lambda_ot
        self.lambda_r      = lambda_r
        self.lambda_M      = lambda_M
        self.lambda_M1     = lambda_M1
        self.lambda_kl     = lambda_kl
        self.lambda_pca    = lambda_pca
        self.lambda_f      = lambda_f
        self.lambda_row_sum = lambda_row_sum
        self._density_criterion = torch.nn.KLDivLoss(reduction="mean")
        self.predictA = predictA

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),      nn.BatchNorm1d(64),  nn.ReLU()
        )
        self.fc_mu     = nn.Linear(64, embedding_dim)
        self.fc_logvar = nn.Linear(64, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, n_genes),       nn.Softplus()
        )
        self.align_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        device = setup_device()
        self.A = nn.Parameter(torch.tensor(
            np.random.normal(0, 1, (n_st_spots, n_sc_cells)),
            device=device, requires_grad=True, dtype=torch.float32))
        self.f         = nn.Parameter(torch.randn(n_sc_cells, 1, device=device))
        self.log_scale = nn.Parameter(torch.tensor(0.0))

        # <<< CHANGED: 注册 loss_threshold buffer，训练前由外部写入 >>>
        # 用 register_buffer 使其随 .to(device) 自动迁移，但不参与梯度
        self.register_buffer('loss_threshold', torch.tensor(float('inf')))

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode_sc(self, X_sc):
        h      = self.encoder(X_sc)
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return self.reparameterize(mu, logvar), mu, logvar

    def decode_expression(self, z_cell):
        return self.decoder(z_cell)

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def compute_softmax_A(self):
        return F.softmax(self.A, dim=0)

    def compute_st_embedding(self, z_cell):
        return torch.mm(self.compute_softmax_A(), z_cell)

    def poisson_nll_loss(self, y_true, lam):
        """
        Robust Poisson NLL。
        - y <= loss_threshold : 标准 Poisson NLL (lam - y*log(lam))
        - y >  loss_threshold : 归一化 L1，截断极端梯度
        loss_threshold 在训练前由 train_vae_poisson_model 预计算并写入，
        避免在每个 forward 里对大矩阵做 quantile（会触发 CUDA 大小限制）。
        """
        lam = torch.clamp(lam, min=1e-8)

        # <<< CHANGED: 用 self.loss_threshold，不再动态 quantile >>>
        threshold    = self.loss_threshold
        mask_normal  = (y_true <= threshold)

        loss_normal  = F.poisson_nll_loss(
            torch.log(lam + 1e-8), y_true,
            log_input=True, full=False, reduction='none'
        )
        loss_outlier = torch.abs(lam - y_true) / (y_true + 1.0)

        return torch.where(mask_normal, loss_normal, loss_outlier).mean()

    def loss_L1(self, X_sc, f_vals):
        return self.poisson_nll_loss(X_sc, f_vals)

    def loss_L1_cos(self, X_sc, f_vals):
        return 1 - 0.5 * cosine_similarity(f_vals, X_sc, dim=0).mean()

    def loss_L2(self, X_st, f_vals, A_softmax=None):
        if self.lambda_f == 0:
            X_st_pred = torch.mm(A_softmax, f_vals)
        else:
            s_vec     = torch.sigmoid(self.f).squeeze()
            weighted  = s_vec.unsqueeze(-1) * f_vals if f_vals.dim() > 1 else s_vec * f_vals
            X_st_pred = torch.mm(A_softmax, weighted)
        return self.poisson_nll_loss(X_st, self.log_scale * X_st_pred)

    def loss_L3(self, z_st, S):
        diff    = z_st.unsqueeze(1) - z_st.unsqueeze(0)
        sq_diff = torch.mean(diff ** 2, dim=2)
        return torch.mean(S * sq_diff) if torch.sum(S) > 0 else torch.sum(sq_diff) * 1e-6

    def loss_LM(self, X_sc, X_st):
        A_softmax = self.compute_softmax_A()
        if self.lambda_f == 0:
            G_pred = torch.matmul(A_softmax, X_sc)
        else:
            weighted = torch.sigmoid(self.f) * X_sc
            G_pred   = torch.matmul(A_softmax, weighted)
        return 1 - 0.5 * cosine_similarity(G_pred, X_st, dim=0).mean()

    def custom_loss(self):
        f_s = torch.sigmoid(self.f)
        return torch.mean(f_s - f_s ** 2)

    def row_sum_mse_loss(self, num_cell):
        A_softmax = self.compute_softmax_A()
        A = A_softmax if self.lambda_f == 0 else A_softmax * torch.sigmoid(self.f).squeeze()
        row_sums = torch.log(A.sum(dim=1) / A.sum())
        num_cell = num_cell / num_cell.sum()
        return self._density_criterion(row_sums.unsqueeze(0), num_cell.unsqueeze(0))

    def forward(self, X_sc, X_st, z_pca=None, S=None, num_cell=None, f_vals=None, z_cell=None):
        target_dtype = next(self.parameters()).dtype

        def ensure_dtype(t):
            return None if t is None else t.to(target_dtype)

        X_sc = ensure_dtype(X_sc)
        X_st = ensure_dtype(X_st)

        if self.predictA is not None:
            S        = ensure_dtype(S)
            num_cell = ensure_dtype(num_cell)
            z_st     = self.compute_st_embedding(z_cell)
            l3       = self.loss_L3(z_st, S)
            LM       = self.loss_LM(X_sc, X_st)
            LM1      = self.loss_LM(f_vals, X_st)
            l_f      = self.custom_loss()
            row_sum_loss = self.row_sum_mse_loss(num_cell) if num_cell is not None else torch.tensor(0.0)
            total = (self.lambda_M * LM + self.lambda_M1 * LM1
                     + self.lambda_f * l_f + self.lambda_row_sum * row_sum_loss
                     + self.lambda_reg * l3)
            return total, LM, LM1, l_f, row_sum_loss, l3
        else:
            z_cell, mu, logvar = self.encode_sc(X_sc)
            f_vals      = self.decode_expression(z_cell)
            l1          = self.loss_L1(X_sc, f_vals)
            kl          = self.kl_loss(mu, logvar)
            l4          = self.loss_L1_cos(X_sc, f_vals)
            if z_pca is None:
                raise ValueError("z_pca must be provided for alignment loss.")
            z_cell_proj = self.align_mlp(z_cell)
            z_pca       = torch.tensor(z_pca, device=X_sc.device, dtype=X_sc.dtype)
            loss_align  = F.mse_loss(z_cell_proj, z_pca, reduction='mean')
            total = l1 + self.lambda_pca * loss_align + self.lambda_kl * kl + self.lambda_l4 * l4
            return total, l1, l4, kl, loss_align


def train_vae_poisson_model(sc_path, st_path, C=None, num_cell=None,
                             output_dir='SpatialVG_VAE_Poisson', common_genes=None,
                             embedding_dim=8, lr=0.01, epochs=300,
                             lambda_reg=0.5, lambda_l2=1, lambda_l4=1, lambda_l5=1,
                             lambda_ot=-1, lambda_M=1, lambda_M1=1, lambda_r=-10000,
                             lambda_kl=0.1, lambda_pca=1, lambda_f=1, lambda_row_sum=1.0,
                             n_top_genes=1000, z_pca=None, predictA=None,
                             f_vals=None, z_cell=None, fix_genes=False):
    device = setup_device()
    os.makedirs(f'{output_dir}/models',     exist_ok=True)
    os.makedirs(f'{output_dir}/evaluation', exist_ok=True)

    X_sc, X_st, Y, adata_sc, adata_ST, genes = load_and_align_data(
        sc_path, st_path, n_top_genes=n_top_genes,
        common_genes=common_genes, fix_genes=fix_genes)

    X_sc = transform_quantile_clip(X_sc, q=0.995)
    X_st = transform_quantile_clip(X_st, q=0.995)

    if predictA is not None:
        S = create_similarity_graph(Y.numpy()) if Y is not None else torch.zeros(X_st.shape[0], X_st.shape[0])
        S = S.to(device)

    # <<< CHANGED: 在移到 GPU 之前，在 CPU 上计算 99% 分位阈值 >>>
    # torch.quantile 对大矩阵在 GPU 上有元素数量限制（约 2^24），
    # CPU 无此限制；算好标量后再 .to(device) 开销极小。
    #sc_threshold = torch.quantile(X_sc.float(), 0.99)   # CPU，返回标量 tensor
    
    sc_threshold = torch.tensor(
        float(np.percentile(X_sc.numpy(), 99)),
        dtype=torch.float32
    )
    print(f"loss_threshold (99th pct of X_sc): {sc_threshold.item():.4f}")

    X_sc, X_st = X_sc.to(device), X_st.to(device)

    model = VAEImprovedSpatialModel(
        n_sc_cells=X_sc.shape[0], n_st_spots=X_st.shape[0],
        n_genes=X_sc.shape[1], embedding_dim=embedding_dim,
        lambda_reg=lambda_reg, lambda_l2=lambda_l2, lambda_l4=lambda_l4,
        lambda_l5=lambda_l5, lambda_ot=lambda_ot, lambda_M=lambda_M,
        lambda_M1=lambda_M1, lambda_r=lambda_r, lambda_kl=lambda_kl,
        lambda_pca=lambda_pca, lambda_f=lambda_f,
        lambda_row_sum=lambda_row_sum, predictA=predictA
    ).to(device)

    # <<< CHANGED: 将预算好的阈值写入 model buffer，随模型自动在正确 device >>>
    model.loss_threshold = sc_threshold.to(device)

    if z_pca is None:
        pca_sc = initialize_assignment_matrix(model, X_sc, X_st, n_neighbors=15)
    else:
        pca_sc = z_pca.copy()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    best_loss, patience = float('inf'), 0
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if predictA is None:
            total, l1, l2, kl, loss_align = model(X_sc, X_st, z_pca=pca_sc[:, :embedding_dim])
        else:
            total, LM, LM1, l_f, row_sum_loss, l3 = model(
                X_sc, X_st, S=S, num_cell=num_cell, f_vals=f_vals, z_cell=z_cell)

        if torch.isnan(total).any():
            print(f"NaN detected at epoch {epoch}. Stopping training.")
            break

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if predictA is None:
            losses.append({'epoch': epoch, 'total': total.item(),
                           'l1': l1.item(), 'kl': kl.item(), 'loss_align': loss_align.item()})
        else:
            losses.append({'epoch': epoch, 'total': total.item(),
                           'LM': LM.item(), 'LM1': LM1.item(), 'l_f': l_f.item(),
                           'row_sum_loss': row_sum_loss.item(), 'l3': l3.item()})

        if total.item() < best_loss:
            best_loss, patience = total.item(), 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'common_genes': genes,
                'config': {
                    'n_sc_cells': X_sc.shape[0], 'n_st_spots': X_st.shape[0],
                    'n_genes': X_sc.shape[1], 'embedding_dim': embedding_dim,
                    'lambda_reg': lambda_reg, 'lambda_l2': lambda_l2,
                    'lambda_l4': lambda_l4, 'lambda_l5': lambda_l5,
                    'lambda_ot': lambda_ot, 'lambda_M': lambda_M,
                    'lambda_r': lambda_r, 'lambda_kl': lambda_kl,
                    'loss_align': lambda_pca, 'loss_f': lambda_f,
                    'lambda_row_sum': lambda_row_sum
                }
            }, f'{output_dir}/models/best_vae_poisson_model.pth')
        else:
            patience += 1
            if patience >= 200:
                break

        if epoch % 20 == 0:
            if predictA is None:
                print(f'Epoch {epoch:3d}: Total={total:.2f}, L1={l1:.2f}, '
                      f'L4={l2:.2f}, KL={kl:.2f}, PCA={loss_align:.2f}')
            else:
                print(f'Epoch {epoch:3d}: Total={total:.2f}, LM={LM:.2f}, '
                      f'LM1={LM1:.2f}, l_f={l_f:.2f}, '
                      f'row_sum_loss={10000 * row_sum_loss:.2f}, l3={l3:.2f}')

    return model, genes, losses


import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform


def build_A_gt_from_spot_to_cells(adata_st, adata_sc, spot_to_cells):
    n_spots = adata_st.n_obs
    n_cells = adata_sc.n_obs
    A_gt = np.zeros((n_spots, n_cells), dtype=np.float16)
    for i, spot_name in enumerate(adata_st.obs_names):
        if spot_name in spot_to_cells:
            for cid in spot_to_cells[spot_name]:
                if 0 <= cid < n_cells:
                    A_gt[i, cid] = 1.0
                else:
                    raise ValueError(f"Cell index {cid} out of range for {n_cells} cells")
    return A_gt


def evaluate_predictions_numpy(
    predictsc, scembeding, X_sc_true,
    predictst=None, X_st_true=None,
    predictA=None, A_gt=None,
    cell_type_labels=None, A_gt_deconv=None, cell_types_gt_order=None
):
    metrics = {}
    if predictsc is not None: predictsc   = np.asarray(predictsc,   dtype=np.float32)
    if X_sc_true is not None: X_sc_true   = np.asarray(X_sc_true,   dtype=np.float32)
    if predictst is not None: predictst   = np.asarray(predictst,   dtype=np.float32)
    if X_st_true is not None: X_st_true   = np.asarray(X_st_true,   dtype=np.float32)
    if predictA  is not None: predictA    = np.asarray(predictA,    dtype=np.float32)
    if A_gt      is not None: A_gt        = np.asarray(A_gt,        dtype=np.float32)

    def mean_pcc(x_true, x_pred):
        row_pccs, col_pccs = [], []
        for i in range(x_true.shape[0]):
            t, p = x_true[i], x_pred[i]
            r = float(pearsonr(t, p)[0]) if (np.std(t) > 1e-8 and np.std(p) > 1e-8) else 0.0
            row_pccs.append(r)
        for j in range(x_true.shape[1]):
            t, p = x_true[:, j], x_pred[:, j]
            r = float(pearsonr(t, p)[0]) if (np.std(t) > 1e-8 and np.std(p) > 1e-8) else 0.0
            col_pccs.append(r)
        return np.mean(row_pccs), np.mean(col_pccs)

    if predictsc is not None and X_sc_true is not None:
        r, c = mean_pcc(X_sc_true, predictsc)
        metrics['predictsc_row_pcc'] = float(r)
        metrics['predictsc_col_pcc'] = float(c)

    if predictst is not None and X_st_true is not None:
        r, c = mean_pcc(X_st_true, predictst)
        metrics['predictst_row_pcc'] = float(r)
        metrics['predictst_col_pcc'] = float(c)

    if predictA is not None and A_gt is not None:
        metrics['assignment_accuracy'] = float(
            np.mean(np.argmax(predictA, axis=1) == np.argmax(A_gt, axis=1)))

    dec_pred_aligned = None
    if predictA is not None and cell_type_labels is not None:
        le         = LabelEncoder()
        labels     = le.fit_transform(cell_type_labels)
        one_hot_sc = np.eye(len(le.classes_))[labels]
        dec_pred_raw = predictA @ one_hot_sc
        dec_pred_raw = dec_pred_raw / dec_pred_raw.sum(axis=1, keepdims=True)
        if A_gt_deconv is not None:
            A_gt_deconv  = np.asarray(A_gt_deconv, dtype=np.float32)
            reorder_idx  = [le.classes_.tolist().index(ct) for ct in cell_types_gt_order]
            dec_pred_aligned = dec_pred_raw[:, reorder_idx]
            pcc_list = []
            for i in range(A_gt_deconv.shape[0]):
                gt_i, pred_i = A_gt_deconv[i], dec_pred_aligned[i]
                r = 0.0
                if np.std(gt_i) > 1e-8 and np.std(pred_i) > 1e-8:
                    r = float(pearsonr(gt_i, pred_i)[0])
                    if not np.isfinite(r): r = 0.0
                pcc_list.append(r)
            metrics['deconv_mean_pcc']  = float(np.mean(pcc_list))
            metrics['deconv_mean_mae']  = float(np.mean(np.abs(A_gt_deconv - dec_pred_aligned)))
            metrics['deconv_mean_rmse'] = float(np.sqrt(np.mean((A_gt_deconv - dec_pred_aligned) ** 2)))
            metrics['PCC_all'] = pcc_list

    if scembeding is not None and cell_type_labels is not None:
        le = LabelEncoder()
        y  = le.fit_transform(cell_type_labels)
        if len(np.unique(y)) > 1:
            dist_mat = squareform(pdist(scembeding, metric='correlation'))
            metrics['silhouette_index'] = float(silhouette_score(dist_mat, y, metric='precomputed'))
        else:
            metrics['silhouette_index'] = 0.0

    print("✅ Evaluation complete.")

    if predictsc is not None and X_sc_true is not None and cell_type_labels is not None:
        cell_type_labels_arr = np.asarray(cell_type_labels)
        unique_types = np.unique(cell_type_labels_arr)[:10]
        n_cols = 5
        n_rows = int(np.ceil(len(unique_types) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        for idx, ct in enumerate(unique_types):
            ax = axes[idx // n_cols][idx % n_cols]
            ct_idx = np.where(cell_type_labels_arr == ct)[0]
            if len(ct_idx) == 0:
                ax.axis("off"); continue
            x, y = X_sc_true[ct_idx[0]], predictsc[ct_idx[0]]
            r = float(pearsonr(x, y)[0]) if (np.std(x) > 1e-8 and np.std(y) > 1e-8) else 0.0
            ax.scatter(x, y, s=6, alpha=0.6)
            ax.set_title(f"{ct}\nPCC={r:.2f}", fontsize=10)
            ax.set_xlabel("X_sc_true"); ax.set_ylabel("predictsc")
        for j in range(idx + 1, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")
        plt.tight_layout(); plt.show()

    return metrics, dec_pred_aligned


# ─────────────────────────────────────────────
# 调用脚本
# ─────────────────────────────────────────────

pcaguide     = pd.read_csv('harmony_embedding.txt', sep='\t', index_col=0)
markers      = np.load('./20k_markers.npy')
common_genes = markers.tolist()
sc_data_path = "./scRNA_subsampled_20k.h5ad"
st_data_path = "./Visium_FAD.h5ad"

count_file_path = './spot_loc_with_counts_r_f.csv'
Count_cell = pd.read_csv(count_file_path)
num_cell   = torch.tensor(Count_cell['n_cells'].values, device='cuda:1')

print("=" * 60)
print("Starting improved model training")
print("=" * 60)

model, common_genes, losses = train_vae_poisson_model(
    sc_path=sc_data_path, st_path=st_data_path,
    C=None, num_cell=num_cell,
    output_dir='SpatialVG_improved_NMF', common_genes=common_genes,
    embedding_dim=32, lr=0.05, epochs=500,
    lambda_reg=0, lambda_l2=0, lambda_l4=2, lambda_l5=0, lambda_ot=0,
    lambda_M=0, lambda_M1=0, lambda_r=0, lambda_kl=0.0001,
    lambda_pca=1, lambda_f=0, lambda_row_sum=0,
    n_top_genes=5000, z_pca=pcaguide.values
)

os.makedirs('SpatialVG_improved_NMF-train-test/models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'common_genes':     common_genes,
    'training_losses':  losses,
}, 'SpatialVG_improved_NMF/models/fix_enc_pca1_top5000_kl_soft_harm-best_result0.5-withmarker2testforzhou.pth')
print("Training completed and model saved.")

# ── Performance evaluation ──
n_top_genes  = 5000
markers      = np.load('./20k_markers.npy')
common_genes = markers.tolist()
device       = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

X_sc, X_st, Y, adata_sc, adata_ST, common_genes = load_and_align_data(
    sc_data_path, st_data_path, n_top_genes, common_genes=common_genes)

# <<< CHANGED: evaluation 与训练用同一变换 >>>
X_sc = transform_quantile_clip(X_sc, q=0.995)
X_st = transform_quantile_clip(X_st, q=0.995)

checkpoint_path = ('SpatialVG_improved_NMF/models/'
                   'fix_enc_pca1_top5000_kl_soft_harm-best_result0.5-withmarker2testforzhou.pth')
checkpoint   = torch.load(checkpoint_path, map_location=device, weights_only=False)
common_genes = checkpoint['common_genes']
adata_sc     = adata_sc[:, common_genes]
adata_st     = adata_ST[:, common_genes]
A_gt_deconv  = pd.read_csv('./S3_GT.txt', sep='\t', header=0, index_col=0)

model.eval()
device      = next(model.parameters()).device
X_sc_tensor = X_sc.to(device).float()

with torch.no_grad():
    z_cell, _, _ = model.encode_sc(X_sc_tensor)
    f_vals        = model.decoder(z_cell)
    A_hat         = model.compute_softmax_A()
    scale         = torch.exp(model.log_scale)
    X_st_pred     = torch.mm(scale * A_hat, f_vals)
    sigmoid_f     = torch.sigmoid(model.f)

predictsc  = f_vals.cpu().numpy()
X_st_pred  = X_st_pred.cpu().numpy()
A_hat_np   = A_hat.cpu().numpy()
scembeding = z_cell.cpu().numpy()

metadata            = adata_sc.obs.copy()
cell_type_labels    = metadata['cell_type'].values
cell_types_gt_order = A_gt_deconv.columns.tolist()

A_hat_np = pd.DataFrame(A_hat_np, index=adata_st.obs_names, columns=adata_sc.obs_names)
A_hat_np = A_hat_np.loc[A_gt_deconv.index, :]

np.save('fix_enc_pca1_top5000_kl_soft_harm_fval.npy',   predictsc)
np.save('fix_enc_pca1_top5000_kl_soft_harm_z_cell.npy', scembeding)

predictsc  = np.load('fix_enc_pca1_top5000_kl_soft_harm_fval.npy')
scembeding = np.load('fix_enc_pca1_top5000_kl_soft_harm_z_cell.npy')
pcaguide   = pd.read_csv('harmony_embedding.txt', sep='\t', index_col=0)

checkpoint   = torch.load(checkpoint_path, map_location=device, weights_only=False)
common_genes = checkpoint['common_genes']

Count_cell = pd.read_csv(count_file_path)
num_cell   = torch.tensor(Count_cell['n_cells'].values, device='cuda:1')

print("=" * 60)
print("Starting second-stage training (A optimisation)")
print("=" * 60)

model, common_genes, losses = train_vae_poisson_model(
    sc_path=sc_data_path, st_path=st_data_path,
    num_cell=num_cell,
    output_dir='SpatialVG_improved_NMF', common_genes=common_genes,
    embedding_dim=32, lr=0.05, epochs=500,
    lambda_reg=1, lambda_l2=0.01, lambda_l4=0, lambda_l5=0, lambda_ot=0,
    lambda_M=0.1, lambda_M1=1, lambda_r=0, lambda_kl=0, lambda_pca=0,
    lambda_f=0.1, lambda_row_sum=100,
    n_top_genes=5000, z_pca=None, predictA=True,
    f_vals=torch.tensor(predictsc,  device='cuda:1'),
    z_cell=torch.tensor(scembeding, device='cuda:1'),
    fix_genes=True
)

os.makedirs('SpatialVG_improved_NMF/models', exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'common_genes':     common_genes,
    'training_losses':  losses,
}, 'SpatialVG_improved_NMF/models/fix_enc_pca1_top5000_kl_soft_harm_getA2forzhou.pth')
print("Training completed and model saved.")

metadata         = adata_sc.obs.copy()
cell_type_labels = metadata['cell_type'].values

model.eval()
device      = next(model.parameters()).device
X_sc_tensor = X_sc.to(device).float()

with torch.no_grad():
    sigmoid_f = torch.sigmoid(model.f).squeeze()
    A_hat     = model.compute_softmax_A()

A_hat_np = A_hat.cpu().numpy()
A_hat_np = pd.DataFrame(A_hat_np, index=adata_st.obs_names, columns=adata_sc.obs_names)
A_hat_np = A_hat_np.loc[A_gt_deconv.index, :]
cell_types_gt_order = A_gt_deconv.columns.tolist()


def get_entropy_uniformity(matrix):
    epsilon = 1e-12
    matrix  = np.clip(matrix, epsilon, 1)
    return -np.sum(matrix * np.log(matrix), axis=0) / np.log(matrix.shape[0])


A_hat_np  = A_hat_np / A_hat_np.sum(axis=0)
AA        = get_entropy_uniformity(A_hat_np)
index0    = np.where(AA < 0.87)[0]
A_hat_np1 = A_hat_np.iloc[:, index0]

result_ours, A_dec_pred = evaluate_predictions_numpy(
    None, None,
    np.log(1 + adata_sc.X.toarray()),
    predictst=None,
    X_st_true=np.log(1 + adata_st.X.toarray()),
    predictA=A_hat_np1,
    A_gt=None,
    cell_type_labels=cell_type_labels[index0],
    A_gt_deconv=A_gt_deconv,
    cell_types_gt_order=cell_types_gt_order
)

A_dec_pred1 = A_dec_pred.copy()
A_dec_pred1 = (A_dec_pred1.T / A_dec_pred1.sum(axis=1)).T
AA          = get_entropy_uniformity(A_dec_pred1.T)
index1      = np.where(AA < 0.90)[0]

pcc_list = []
for i in range(A_gt_deconv.iloc[index1, :].shape[0]):
    gt_i, pred_i = A_gt_deconv.iloc[index1, :].values[i], A_dec_pred1[index1, :][i]
    r = 0.0
    if np.std(gt_i) > 1e-8 and np.std(pred_i) > 1e-8:
        r = float(pearsonr(gt_i, pred_i)[0])
        if not np.isfinite(r): r = 0.0
    pcc_list.append(r)


def generate_optimized_A_softmax_normalized(A0, adj_matrix, lambda_reg=10.0, lr=0.05, epochs=1000):
    device     = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    A0_tensor  = torch.tensor(A0, dtype=torch.float32).to(device)
    W          = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    D_vec      = W.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_vec + 1e-8))
    I          = torch.eye(W.shape[0]).to(device)
    L_sym      = I - torch.mm(torch.mm(D_inv_sqrt, W), D_inv_sqrt)
    S          = torch.nn.Parameter(torch.log(A0_tensor + 1e-8) + torch.randn_like(A0_tensor) * 0.01)
    optimizer  = torch.optim.Adam([S], lr=lr)
    print(f"开始优化 (归一化拉普拉斯, Device: {device})...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        A           = torch.softmax(S, dim=1)
        norm_A      = torch.clamp(A.norm(dim=1, keepdim=True),  min=1e-8)
        norm_A0     = torch.clamp(A0_tensor.norm(dim=1, keepdim=True), min=1e-8)
        cos_sim     = (A * A0_tensor).sum(dim=1, keepdim=True) / (norm_A * norm_A0)
        loss_sim    = (1 - cos_sim).mean()
        loss_smooth = torch.trace(A.t() @ L_sym @ A)
        loss        = loss_sim + lambda_reg * loss_smooth
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                  f"Sim: {loss_sim.item():.4f}, Smooth: {loss_smooth.item():.4f}")
    return torch.softmax(S, dim=1).detach().cpu().numpy()


Count_cell  = pd.read_csv(count_file_path, index_col=0).loc[A_gt_deconv.index, :]
coords      = Count_cell[['imagerow', 'imagecol']].values

knn = NearestNeighbors(n_neighbors=5, metric='euclidean').fit(coords)
distances, indices = knn.kneighbors(coords)

adj_matrix = np.zeros((len(coords), len(coords)))
for i in range(len(coords)):
    adj_matrix[i, indices[i]] = 1
adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
print("邻接矩阵形状:", adj_matrix.shape, "  非零元素:", np.count_nonzero(adj_matrix))

A_optimized = generate_optimized_A_softmax_normalized(
    A_dec_pred1.copy(), adj_matrix.copy(), lambda_reg=0.01, epochs=1000)
print("优化完成，A 的形状:", A_optimized.shape)

pcc_list = []
for i in range(A_gt_deconv.shape[0]):
    gt_i, pred_i = A_gt_deconv.values[i], A_optimized[i]
    r = 0.0
    if np.std(gt_i) > 1e-8 and np.std(pred_i) > 1e-8:
        r = float(pearsonr(gt_i, pred_i)[0])
        if not np.isfinite(r): r = 0.0
    pcc_list.append(r)
print(float(np.mean(pcc_list)))

pd.DataFrame(pcc_list).to_csv('pcc_list_2026_4_29.csv')
import csv
with open('pcc_list_oursbest1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['PCC'])          # 写入表头
    for r in pcc_list:
        writer.writerow([r])


