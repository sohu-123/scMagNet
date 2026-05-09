import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import umap
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import QuantileTransformer

# 禁用 PIL 的解压炸弹限制（允许大图像）
Image.MAX_IMAGE_PIXELS = None

# 创建输出目录
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 所有输入文件根目录
data_dir = "./dataGithub"

# ============================================================
# 读取数据
# ============================================================
data0 = np.load(os.path.join(data_dir, 'jiyuanyang_marker_stembeding_filter.npy'))
data1 = np.load(os.path.join(data_dir, 'jiyuanyang_marker_scembeding_filter.npy'))
data_combined = np.vstack((data0, data1))

st_data_path = os.path.join(data_dir, "Visium_FAD.h5ad")
adata_st = sc.read_h5ad(st_data_path)

A_gt_deconv = pd.read_csv(os.path.join(data_dir, 'S3_GT.txt'), sep='\t', header=0, index_col=0)

sc_data_path = os.path.join(data_dir, "scRNA_subsampled_20k.h5ad")
adata_sc = sc.read_h5ad(sc_data_path)

x = adata_st.obsm['spatial'][:, 1]

# ============================================================
# UMAP 图 1: data_combined + labels (原代码未用该图，但保留计算)
# ============================================================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_comb = reducer.fit_transform(data_combined)
labels = np.concatenate([np.zeros(len(data0)), np.ones(len(data1))]).astype(int)

# ============================================================
# UMAP 图: data0 按连续变量 x 着色
# ============================================================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(data0)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=x, cmap='viridis', s=10, alpha=0.8)
plt.colorbar(scatter, label='Value of x')
plt.title('UMAP Projection Colored by Continuous Variable x')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'umap_continuous_x.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# dev_A 相关
# ============================================================
dev_A = np.load(os.path.join(data_dir, 'A_dec_pred-best_by_2026-04-07.npy'))
dev_A1 = pd.DataFrame(dev_A, index=A_gt_deconv.index, columns=A_gt_deconv.columns)
dev_A1.to_csv(os.path.join(output_dir, 'A_dec_pred-best_by_2026-04-07.csv'), index=True, header=True)

count_file_path = os.path.join(data_dir, 'spot_loc_with_counts_r_f.csv')
Count_cell = pd.read_csv(count_file_path)
Count_cell = Count_cell.set_index(['Unnamed: 0'])
Count_cell = Count_cell.loc[A_gt_deconv.index, :]

# ============================================================
# UMAP 图: data0 按预测标签 y_pred 着色
# ============================================================
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(data0)
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=adata_st.obs_names)
embedding_df = embedding_df.loc[A_gt_deconv.index, :]
embedding = embedding_df.values
y_pred = dev_A.argmax(axis=1)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab20.colors + plt.cm.Set3.colors
colors = colors[:27]
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred,
                      cmap=plt.matplotlib.colors.ListedColormap(colors), s=10, alpha=0.8)
plt.colorbar(scatter, label='Predicted Label')
plt.title('UMAP Projection Colored by Predicted Label')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'umap_predicted_labels.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 循环绘制 dev_A 每列的圆形图 (第一组，自定义颜色映射)
# ============================================================
img_path = os.path.join(data_dir, "VisiumImage_FAD_1.jpg")
img = np.array(Image.open(img_path))
print("VisiumImage_FAD_1.jpg shape:", img.shape)

x_coords = Count_cell['imagecol'].values
y_coords = Count_cell['imagerow'].values
r = 73.1/2 + 15
custom_cmap = mcolors.LinearSegmentedColormap.from_list('gray_to_green', ['#2C3539', '#00FA9A'])
num_cols = dev_A.shape[1]

for i in range(int(num_cols/6)):
    gene_expression = dev_A[:, i]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, interpolation='bilinear')
    patches = [Circle((xi, yi), r) for xi, yi in zip(x_coords, y_coords)]
    p = PatchCollection(patches, edgecolor=None, linewidth=0.0001, rasterized=False)
    p.set_array(np.array(gene_expression))
    p.set_cmap(custom_cmap)
    ax.add_collection(p)
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Expression Level', fontsize=12)
    ax.set_title(f"{A_gt_deconv.columns[i]}", fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"deconv_result_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 循环绘制 A_gt_deconv 每列的圆形图 (第二组，viridis 颜色映射)
# ============================================================
for i in range(int(num_cols/6)):
    gene_expression = A_gt_deconv.values[:, i]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, interpolation='bilinear')
    patches = [Circle((xi, yi), r) for xi, yi in zip(x_coords, y_coords)]
    p = PatchCollection(patches, edgecolor=None, linewidth=0.0001, rasterized=False)
    p.set_array(np.array(gene_expression))
    p.set_cmap('viridis')
    ax.add_collection(p)
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Expression Level', fontsize=12)
    ax.set_title(f"{A_gt_deconv.columns[i]}", fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"deconv_result_gt_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# SCC 数据部分
# ============================================================
A_optimized_df = pd.read_csv(os.path.join(data_dir, 'A_optimized_scc-spotiphy.csv'), sep=',', header=0, index_col=0)
sc_path = os.path.join(data_dir, "adata_SCC_sc.h5ad")
st_path = os.path.join(data_dir, "adata_SCC_ST.h5ad")
adata_st = sc.read(st_path)
dev_A = A_optimized_df
A_gt_deconv = A_optimized_df

img_path = os.path.join(data_dir, "GSM4284316_P2_ST_rep1.jpg")
img = np.array(Image.open(img_path))
print("SCC image shape:", img.shape)

x_coords = adata_st.obs['pixel_x'].values
y_coords = adata_st.obs['pixel_y'].values
r = 100
num_cols = dev_A.shape[1]

for i in range(int(num_cols/4)):
    gene_expression = A_gt_deconv.values[:, i]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, interpolation='bilinear')
    patches = [Circle((xi, yi), r) for xi, yi in zip(x_coords, y_coords)]
    p = PatchCollection(patches, edgecolor=None, linewidth=0.0001, rasterized=False)
    p.set_array(np.array(gene_expression))
    p.set_cmap('viridis')
    ax.add_collection(p)
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Expression Level', fontsize=12)
    ax.set_title(f"{A_gt_deconv.columns[i]}", fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scc_deconv_result_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# level2
A_optimized_df = pd.read_csv(os.path.join(data_dir, 'A_optimized_scc-spotiphy-level2.csv'), sep=',', header=0, index_col=0)
dev_A = A_optimized_df
A_gt_deconv = A_optimized_df
num_cols = dev_A.shape[1]
for i in range(int(num_cols/4)):
    gene_expression = A_gt_deconv.values[:, i]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, interpolation='bilinear')
    patches = [Circle((xi, yi), r) for xi, yi in zip(x_coords, y_coords)]
    p = PatchCollection(patches, edgecolor=None, linewidth=0.0001, rasterized=False)
    p.set_array(np.array(gene_expression))
    p.set_cmap('viridis')
    ax.add_collection(p)
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Expression Level', fontsize=12)
    ax.set_title(f"{A_gt_deconv.columns[i]}", fontsize=14, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scc_level2_deconv_result_{i}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# MERFISH AUC 计算部分
# ============================================================
def calculate_metrics_with_curves(A_gt, A_pred):
    y_true_flat = A_gt.flatten()
    y_score_flat = A_pred.flatten()
    try:
        global_auc = roc_auc_score(y_true_flat, y_score_flat)
        global_auprc = average_precision_score(y_true_flat, y_score_flat)
    except ValueError as e:
        print(f"Global 指标计算出错: {e}")
        global_auc, global_auprc = 0.0, 0.0
    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
    precision, recall, _ = precision_recall_curve(y_true_flat, y_score_flat)
    max_len = max(len(fpr), len(recall))
    curves_df = pd.DataFrame({
        'ROC_FPR': fpr if len(fpr) == max_len else np.append(fpr, [np.nan] * (max_len - len(fpr))),
        'ROC_TPR': tpr if len(tpr) == max_len else np.append(tpr, [np.nan] * (max_len - len(tpr))),
        'PR_Recall': recall if len(recall) == max_len else np.append(recall, [np.nan] * (max_len - len(recall))),
        'PR_Precision': precision if len(precision) == max_len else np.append(precision, [np.nan] * (max_len - len(precision)))
    })
    if A_gt.ndim == 1:
        A_gt = A_gt.reshape(-1, 1)
        A_pred = A_pred.reshape(-1, 1)
    col_aucs = []
    col_auprcs = []
    num_cols = A_gt.shape[1]
    for i in range(num_cols):
        y_true_col = A_gt[:, i]
        y_score_col = A_pred[:, i]
        if len(np.unique(y_true_col)) < 2:
            col_aucs.append(np.nan)
            col_auprcs.append(np.nan)
        else:
            col_aucs.append(roc_auc_score(y_true_col, y_score_col))
            col_auprcs.append(average_precision_score(y_true_col, y_score_col))
    mean_col_auc = np.nanmean(col_aucs)
    mean_col_auprc = np.nanmean(col_auprcs)
    return global_auc, global_auprc, col_aucs, col_auprcs, curves_df

tang_ad_map = sc.read(os.path.join(data_dir, "tangram_mapping-Simulation.h5ad"))
A_gt = np.load(os.path.join(data_dir, 'groundtruth_A_m1s1s50.npy'))
ours = pd.read_csv(os.path.join(data_dir, 'predicted_A_matrix-0425-BestModelLM1L30.1_novo_0.0001.csv'), sep=',', index_col=0)
novo = np.load(os.path.join(data_dir, 'A_SimulationData-alpha0.5-epsilon-3-2026-04-22.npy'))

A_pred_novo = novo.T
g_auc, g_auprc, novo_m_auc, novo_m_auprc, novo_df = calculate_metrics_with_curves(A_gt[:, A_gt.sum(axis=0)>0], A_pred_novo[:, A_gt.sum(axis=0)>0])
novo_df.to_csv(os.path.join(output_dir, 'merfish_novo_curve.csv'))

A_pred_ours = ours.values
g_auc, g_auprc, o_m_auc, o_m_auprc, o_df = calculate_metrics_with_curves(A_gt[:, A_gt.sum(axis=0)>0], A_pred_ours[:, A_gt.sum(axis=0)>0])
o_df.to_csv(os.path.join(output_dir, 'merfish_o_curve.csv'))

A_pred_tang = tang_ad_map.X.T
g_auc, g_auprc, t_m_auc, t_m_auprc, t_df = calculate_metrics_with_curves(A_gt[:, A_gt.sum(axis=0)>0], A_pred_tang[:, A_gt.sum(axis=0)>0])
t_df.to_csv(os.path.join(output_dir, 'merfish_t_curve.csv'))

df_auc = pd.DataFrame({'novo': novo_m_auc, 'ours': o_m_auc, 'tangram': t_m_auc})
df_pr = pd.DataFrame({'novo': novo_m_auprc, 'ours': o_m_auprc, 'tangram': t_m_auprc})
df_auc.to_csv(os.path.join(output_dir, 'merfish_col_auc.csv'), index=False)
df_pr.to_csv(os.path.join(output_dir, 'merfish_col_aupr.csv'), index=False)

# 部分细胞类型筛选
data0_mer = pd.read_csv(os.path.join(data_dir, 'merge_df_mouse1sample1_mouse1_slice50.csv'), sep=',', header=0)
index0 = data0_mer['subclass'].values == 'L2/3 IT'
tang_ad_map_sub = tang_ad_map[index0, :]
ours_sub = ours.values[:, index0]
A_gt_sub = A_gt[:, index0]
novo_sub = novo[index0, :]

# 分位数归一化
qt = QuantileTransformer(output_distribution='uniform', random_state=42)
A_normalized = qt.fit_transform(ours)
A_normalized = pd.DataFrame(A_normalized, index=ours.index, columns=ours.columns)
A_normalized.to_csv(os.path.join(output_dir, 'merfish_A_quantile.csv'))

pd.DataFrame(data0_mer['subclass'].values[A_gt.sum(axis=0)>0]).to_csv(os.path.join(output_dir, 'merfish_cell_type_inspot.csv'))

# ============================================================
# 模拟数据解卷积比较
# ============================================================
data0_sim = pd.read_csv(os.path.join(data_dir, 'merge_df_mouse1sample1_mouse1_slice50.csv'), sep=',', header=0)
A_gt_deconv_sim = pd.read_csv(os.path.join(data_dir, 'A_gt_simulationdata.csv'), sep=',', header=0, index_col=0)
spatial_with_tangram = sc.read(os.path.join(data_dir, "spatial_with_tangram-Simulation.h5ad"))
tangram = spatial_with_tangram.obsm["tangram_ct_pred"]
spotiphy = pd.read_csv(os.path.join(data_dir, 'proportion.csv'), header=None)
novo_sim = np.load(os.path.join(data_dir, 'A_SimulationData-alpha0.5-epsilon-3-2026-04-22.npy'))

subclass = data0_sim['subclass'].values
df_onehot = pd.get_dummies(subclass)
X = df_onehot.values.T
result = np.dot(X, novo_sim)

tangram = tangram[A_gt_deconv_sim.columns]
pcc_values_tan = np.zeros(424)
X_mat = A_gt_deconv_sim.values
Y_tan = tangram.values
for i in range(424):
    pcc_values_tan[i] = np.corrcoef(X_mat[i, :], Y_tan[i, :])[0,1]

pcc_values_novo = np.zeros(424)
Y_novo = result.T
for i in range(424):
    pcc_values_novo[i] = np.corrcoef(X_mat[i, :], Y_novo[i, :])[0,1]

pcc_values_s = np.zeros(424)
Y_s = spotiphy.values
for i in range(424):
    pcc_values_s[i] = np.corrcoef(X_mat[i, :], Y_s[i, :])[0,1]

ours_pcc = pd.read_csv(os.path.join(data_dir, 'pcc_list_0.5-best_by_2026-04-27-version2-simulationdata1-BestModel0.csv'))
result_pcc = pd.DataFrame({'Spotiphy': pcc_values_s, 'Trangram': pcc_values_tan, 'NovoSpaRc': pcc_values_novo, 'Ours': ours_pcc['0'].values})
result_pcc.to_csv(os.path.join(output_dir, 'PCC_deconv.csv'))
print(result_pcc.mean(axis=0))

# ============================================================
# 空间基因表达图函数
# ============================================================
def plot_spatial_expression(stdata, genes_to_plot, output_name):
    existing_genes = [g for g in genes_to_plot if g in stdata.var_names]
    if not existing_genes:
        print(f"警告：没有找到任何基因，跳过 {output_name}")
        return
    # 坐标变换
    A = stdata.obsm['spatial'].copy()
    A = A[:, [1, 0]]
    A[:, 1] = -A[:, 1]
    stdata.obsm['spatial'] = A
    x_coords = stdata.obsm['spatial'][:, 0]
    y_coords = stdata.obsm['spatial'][:, 1]
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    max_range = max(x_range, y_range)
    x_center = (x_coords.max() + x_coords.min()) / 2
    y_center = (y_coords.max() + y_coords.min()) / 2
    x_lim = [x_center - max_range/2, x_center + max_range/2]
    y_lim = [y_center - max_range/2, y_center + max_range/2]
    n_genes = len(existing_genes)
    n_cols = min(5, n_genes)
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_genes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for idx, gene in enumerate(existing_genes):
        ax = axes[idx]
        expression = stdata[:, gene].X.toarray().flatten() if hasattr(stdata[:, gene].X, 'toarray') else stdata[:, gene].X.flatten()
        scat = ax.scatter(stdata.obsm['spatial'][:, 0], stdata.obsm['spatial'][:, 1], c=expression, cmap='viridis', s=20, alpha=0.8, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(gene, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Expression', fontsize=10)
    for idx in range(len(existing_genes), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{output_name}.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{output_name}.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"已保存 {output_name}.pdf 和 .png")

# FAD 数据 Cplx3
stdata_fad = sc.read_h5ad(os.path.join(data_dir, "Visium_FAD.h5ad"))
plot_spatial_expression(stdata_fad, ['Cplx3'], 'spatial_gene_expression_FAD_Cplx3')

# WT 数据第一组
stdata_wt = sc.read_h5ad(os.path.join(data_dir, "st_10x_WT.h5ad"))
plot_spatial_expression(stdata_wt, ["Rorb", "Cplx3", "Aqp4", 'Pvalb', 'Camk2b'], 'spatial_gene_expression_WT')

# WT 数据第二组
plot_spatial_expression(stdata_wt, ["Egr3", "Mef2c", "Zbtb33(+)", "Foxf2(+)"], 'spatial_gene_expression_WT2')

print("所有图片已保存到 ./output 目录，所有数据读取自 ./dataGithub 目录。")
