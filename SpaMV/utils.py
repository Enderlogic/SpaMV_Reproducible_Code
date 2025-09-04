import math
import os.path

import anndata
import matplotlib.lines as mlines
import numpy as np
import pandas
import pandas as pd
import scanpy as sc
import torch
from matplotlib import pyplot as plt, ticker
from natsort import natsorted
from numpy.linalg import norm
from pandas import DataFrame
from scanpy.plotting import embedding, spatial
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import coalesce
from squidpy.pl import spatial_scatter
from scipy.stats import pearsonr
import seaborn as sns

def construct_graph_by_coordinate(cell_position, neighborhood_depth=3, device='cpu'):
    '''
    Constructing spatial neighbor graph according to spatial coordinates.
    Args:
        cell_position: ndarray of shape (n_cells, 2)
        neighborhood_depth: The Neighborhood Depth parameter determines the number of layers of neighbors to consider
                            when calculating the neighborhood of each node in a network.
                            When set to 1, only the node itself is considered.
                            When set to 2, the node and its immediate neighbors (those directly connected to it) are included.
                            When set to 3, the node, its immediate neighbors, and the neighbors of those immediate neighbors (second-degree neighbors) are considered.
    Returns:
        ndarray of shape (n_edges, 2)
    '''
    dist_sort = np.sort(
        np.unique((cell_position[:, 0] - cell_position[0, 0]) ** 2 + (cell_position[:, 1] - cell_position[0, 1]) ** 2))
    return torch.tensor([x for xs in [[[j, i] for j in np.where(
        (cell_position[:, 0] - cell_position[i, 0]) ** 2 + (cell_position[:, 1] - cell_position[i, 1]) ** 2 < dist_sort[
            neighborhood_depth])[0]] for i in range(cell_position.shape[0])] for x in xs], device=device).T


def construct_graph_by_feature(adata, n_neighbors=20, mode="connectivity", metric="correlation", device="cpu"):
    """Constructing feature neighbor graph according to expresss profiles"""
    if 'X_lsi' in adata.obsm:
        feature_graph = torch.tensor(
            kneighbors_graph(adata.obsm['X_lsi'], n_neighbors, mode=mode, metric=metric).todense(), device=device)
    elif 'X_pca' in adata.obsm:
        feature_graph = torch.tensor(
            kneighbors_graph(adata.obsm['X_pca'], n_neighbors, mode=mode, metric=metric).todense(), device=device)
    else:
        feature_graph = torch.tensor(
            kneighbors_graph(adata.X.toarray() if issparse(adata.X) else adata.X, n_neighbors, mode=mode,
                             metric=metric).todense(), device=device)
    return feature_graph.nonzero().t().contiguous()


def adjacent_matrix_preprocessing(adata, neighborhood_depth=3, neighborhood_embedding=20, device="cpu"):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    ######################################## construct spatial graph ########################################
    edge_index = []
    for data in adata:
        edge_index_spatial = construct_graph_by_coordinate(data.obsm['spatial'], neighborhood_depth, device)

        ######################################## construct feature graph ########################################
        if neighborhood_embedding > 0:
            edge_index_feature = construct_graph_by_feature(data, n_neighbors=neighborhood_embedding, device=device)
            edge_index.append(coalesce(torch.cat([edge_index_feature, edge_index_spatial], dim=1)))
        else:
            edge_index.append(edge_index_spatial)
    return edge_index


def get_init_bg(x):
    bg_init = []
    for data in x:
        data = data / data.sum(axis=1, keepdims=True)
        means = data.nanmean(dim=0)
        bg_init.append((means + 1e-15).log())
    return bg_init


def remove_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
        
def plot_embedding_results(adatas, omics_names, topic_abundance, feature_topics, save=True, folder_path=None,
                           file_name=None, show=False, full=False, corresponding_features=True, size=350, crop_coord=None, rb=True):
    element_names = []
    for omics_name in omics_names:
        if omics_name == "Transcriptomics" or "H3K27" in omics_name:
            element_names.append("Gene")
        elif omics_name == "Proteomics":
            element_names.append("Protein")
        elif omics_name == "Epigenomics":
            element_names.append("Region of open chromatin")
        elif omics_name == "Metabolomics":
            element_names.append("Metabolite")
    zs_dim = len([item for item in topic_abundance.columns if 'Shared' in item])
    n_omics = len(adatas)
    zp_dims = []
    for i in range(n_omics):
        zp_dims.append(len([item for item in topic_abundance.columns if omics_names[i] in item]))
    n_col = max(zp_dims + [zs_dim])
    n_row = n_omics * 3 + 1 if corresponding_features else n_omics + 1
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    if zs_dim < n_col:
        for i in range(n_col - zs_dim):
            for j in range(1 + n_omics if corresponding_features else 1):
                axes[j, zs_dim + i].axis('off')
    for i, zp_dim in enumerate(zp_dims):
        if zp_dim < n_col:
            for j in range(n_col - zp_dim):
                for k in range(2 if corresponding_features else 1):
                    axes[1 + (i + 1) * n_omics + k if corresponding_features else 1 + i + k, zp_dim + j].axis('off')
    adatas[0].obs[topic_abundance.columns] = topic_abundance
    adatas[1].obs[topic_abundance.columns] = topic_abundance
    for i in range(zs_dim):
        topic_name = topic_abundance.columns[i]
        if 'spatial' not in adatas[0].uns:
            embedding(adatas[0], color=topic_name, vmax='p99', size=size, show=False, basis='spatial', ax=axes[0, i])
        else:
            spatial_scatter(adatas[0], color=topic_name, ax=axes[0, i], crop_coord=crop_coord)
        if corresponding_features:
            for j in range(n_omics):
                mrf = feature_topics[omics_names[j]].nlargest(1, topic_name).index[0]
                if 'spatial' not in adatas[j].uns:
                    embedding(adatas[j], color=mrf, vmax='p99', basis='spatial', size=size, cmap='coolwarm', show=False,
                            ax=axes[1 + j, i],
                            title=mrf + '\nMost relevant ' + element_names[j] + '\nw.r.t. ' + topic_name)
                else:
                    # spatial(adatas[j], color=mrf, vmax='p99', cmap='coolwarm', show=False, ax=axes[1 + j, i], title=mrf + '\nMost relevant ' + element_names[j] + '\nw.r.t. ' + topic_name)
                    spatial_scatter(adatas[j], color=mrf, ax=axes[1 + j, i], cmap='coolwarm', crop_coord=crop_coord, title=mrf + '\nMost relevant ' + element_names[j] + '\nw.r.t. ' + topic_name)
    for i in range(zs_dim):
        for j in range(n_omics + 1 if corresponding_features else 1):
            axes[j, i].set_xlabel('')  # Remove x-axis label
            axes[j, i].set_ylabel('')  # Remove y-axis label
            axes[j, i].set_xticks([])  # Remove x-axis ticks
            axes[j, i].set_yticks([])  # Remove y-axis ticks
            if rb:
                remove_box(axes[j, i])
    for i in range(n_omics):
        for j in range(zp_dims[i]):
            topic_name = feature_topics[omics_names[i]].columns[zs_dim + j]
            if 'spatial' not in adatas[0].uns: 
                embedding(adatas[0], color=topic_name, vmax='p99', size=size, show=False, basis='spatial',
                        ax=axes[1 + n_omics + i * n_omics if corresponding_features else 1 + i, j])
            else:
                # spatial(adatas[0], color=topic_name, vmax='p99', show=False, ax=axes[1 + n_omics + i * n_omics if corresponding_features else 1 + i, j])
                spatial_scatter(adatas[0], color=topic_name, ax=axes[1 + n_omics + i * n_omics if corresponding_features else 1 + i, j], crop_coord=crop_coord)
            if corresponding_features:
                mrf = feature_topics[omics_names[i]].nlargest(1, topic_name).index[0]
                if 'spatial' not in adatas[i].uns:
                    embedding(adatas[i], color=mrf, vmax='p99', size=size, show=False, cmap='coolwarm', basis='spatial',
                            title=mrf + '\nMost relevant ' + element_names[i] + '\nw.r.t. ' + topic_name,
                            ax=axes[1 + n_omics + i * n_omics + 1, j])
                else:
                    # spatial(adatas[i], color=mrf, vmax='p99', cmap='coolwarm', show=False, ax=axes[1 + n_omics + i * n_omics + 1, j], title=mrf + '\nMost relevant ' + element_names[i] + '\nw.r.t. ' + topic_name)
                    spatial_scatter(adatas[i], color=mrf, cmap='coolwarm', ax=axes[1 + n_omics + i * n_omics + 1, j], crop_coord=crop_coord, title=mrf + '\nMost relevant ' + element_names[i] + '\nw.r.t. ' + topic_name)
    for i in range(n_omics):
        for j in range(zp_dims[i]):
            if corresponding_features:
                for k in range(2):
                    axes[1 + n_omics * (i + 1) + k, j].set_xlabel('')
                    axes[1 + n_omics * (i + 1) + k, j].set_ylabel('')
                    axes[1 + n_omics * (i + 1) + k, j].set_xticks([])
                    axes[1 + n_omics * (i + 1) + k, j].set_yticks([])
                    if rb:
                        remove_box(axes[1 + n_omics * (i + 1) + k, j])
            else:
                axes[1 + i, j].set_xlabel('')
                axes[1 + i, j].set_ylabel('')
                axes[1 + i, j].set_xticks([])
                axes[1 + i, j].set_yticks([])
                if rb:
                    remove_box(axes[1 + i, j])
    plt.tight_layout()
    if save:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path + 'spamv.pdf' if file_name is None else folder_path + file_name)
    if show:
        plt.show()
    plt.close()
    if full:
        for i in range(n_omics):
            for topic in feature_topics[i].columns:
                embedding(adatas[i], color=[topic] + feature_topics[i].nlargest(8, topic).index.tolist(),
                          basis='spatial', size=size, show=False, ncols=3, vmax='p99')
                fn = folder_path + omics_names[i] + '_' + topic + '.pdf'
                plt.savefig(fn)
                plt.close()


def replace_legend(legend_texts):
    for legend_text in legend_texts:
        if legend_text.get_text() == 'NA':
            legend_text.set_text('other topics')


def plot_clustering_results(adata, cluster_name, omics_names, folder_path, show=False, suffix=None):
    max_width = max([len(item) for item in adata.obs[cluster_name].unique()])
    max_height = adata.obs[cluster_name].nunique()
    max_private_topics = 0
    for omics_name in omics_names:
        max_private_topics = max(max_private_topics,
                                 len([item for item in adata.obs[cluster_name].unique() if omics_name in item]))
    width = 5 + max_width * .12
    height = 4 if max_height < max_private_topics + 1 else 4 + (max_height - max_private_topics - 1) * .2
    adata.obs[cluster_name] = pandas.Categorical(values=adata.obs[cluster_name].values,
                                                 categories=natsorted(adata.obs[cluster_name].unique()), ordered=True)
    adata.obs[cluster_name] = adata.obs[cluster_name].cat.reorder_categories(
        [item for item in natsorted(adata.obs[cluster_name].unique()) if 'Shared' not in item] + [item for item in
                                                                                                  natsorted(adata.obs[
                                                                                                                cluster_name].unique())
                                                                                                  if 'Shared' in item])
    import seaborn as sns
    higher_class_palette = sns.color_palette("Set1", 3)
    high_classes = ['Shared'] + omics_names
    higher_class_colors = {i: color for i, color in zip(high_classes, higher_class_palette)}
    subclass_colors = {}
    for higher_class, color in higher_class_colors.items():
        subclass_labels = [item for item in adata.obs[cluster_name].cat.categories if higher_class in item]
        subclass_palette = sns.light_palette(color, n_colors=len(subclass_labels) + 2)
        for label, subclass_color in zip(subclass_labels, subclass_palette[2:]):
            subclass_colors[label] = subclass_color
    adata.uns['colors'] = [subclass_colors[label] for label in adata.obs[cluster_name].cat.categories]
    for group_name in ['All', 'Shared'] + omics_names:
        if group_name == 'All':
            groups = adata.obs[cluster_name].unique()
        else:
            groups = [item for item in adata.obs[cluster_name].unique() if group_name in item]
        fig, ax = plt.subplots(figsize=(width, height))
        sc.pl.embedding(adata, color=cluster_name, basis='spatial', vmax='p99', size=100,
                        title=group_name + ' topics' if group_name in ['All',
                                                                       'Shared'] else group_name + ' private topics',
                        show=False, groups=groups, ax=ax, palette=adata.uns['colors'])
        plt.plot([], [], label=' ' * (max_width + 19))
        handles, labels = ax.get_legend_handles_labels()
        handles[-1] = mlines.Line2D([], [], color='white')
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.48), frameon=False)
        replace_legend(ax.get_legend().get_texts())
        plt.tight_layout()
        plt.savefig(
            folder_path + group_name + '_topics.pdf' if suffix is None else folder_path + group_name + '_topics_' + suffix + '.pdf')
        if show:
            plt.show()


def ST_preprocess(adata_st, normalize=True, log=True, prune=False, highly_variable_genes=True, n_top_genes=3000, pca=False, n_comps=50, scale=True):
    adata = adata_st.copy()
    if adata.n_vars > 50000:
        sc.pp.filter_genes(adata, min_cells=round(adata.n_obs * .05))

    adata.var['mt'] = np.logical_or(adata.var_names.str.startswith('MT-'), adata.var_names.str.startswith('mt-'))
    adata.var['rb'] = adata.var_names.str.startswith(('RP', 'Rp', 'rp'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < 100
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])
    adata = adata[mask_cell, mask_gene]

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_genes=200)
    if prune:
        adata = adata[:, (adata.X > 1).sum(0) > adata.n_obs / 100]

    if highly_variable_genes:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes, subset=False)
        
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log:
        sc.pp.log1p(adata)

    if pca:
        sc.pp.pca(adata, n_comps=n_comps)

    if scale:
        sc.pp.scale(adata)

    if highly_variable_genes:
        return adata[:, adata.var.highly_variable]
    else:
        return adata


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps, random_state=0)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0,
               increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """

    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key,
                                       n_comps=n_comps if adata.obsm[key].shape[1] > n_comps else adata.obsm[key].shape[
                                           1])

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end,
                             increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))


def compute_similarity(z, w=None):
    similarity_spot = DataFrame(np.zeros((z.shape[1], z.shape[1])), columns=z.columns, index=z.columns)
    for i in z.columns:
        zi = z[i]
        for j in z.columns[np.where(z.columns == i)[0][0] + 1:]:
            zj = z[j]
            similarity_spot.loc[i, j] = cosine_similarity(zi.values, zj.values)

    if w is not None:
        similarity_feature = DataFrame(np.zeros((z.shape[1], z.shape[1])), columns=z.columns, index=z.columns)
        for wi in w:
            for i in wi.columns[:-1]:
                for j in wi.columns[np.where(wi.columns == i)[0][0] + 1:]:
                    similarity_feature.loc[i, j] += cosine_similarity(wi[i], wi[j]) / 2 if i in z.columns and j in z.columns else cosine_similarity(wi[i], wi[j])
        return similarity_spot, similarity_feature
    else:
        return similarity_spot


def visualize_latent(z, location):
    data = anndata.AnnData(z)
    data.obsm['spatial'] = location
    data.obsm['emb'] = z
    clustering(data, key='emb', add_key='emb', n_clusters=5)
    sc.pl.embedding(data, color='emb', basis='spatial')


def compute_gene_topic_correlations(adata, z):
    """
    Compute Pearson correlation between genes and learned topics.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    z : pandas.DataFrame
        Topic matrix where rows are cells and columns are topics
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing correlation coefficients between genes and topics
        Shape: (n_genes, n_topics)
    """
    # Validate input data
    if not isinstance(z, pd.DataFrame):
        try:
            z = pd.DataFrame(z)
        except:
            raise ValueError("z must be convertible to a pandas DataFrame")
    
    # Ensure z contains numerical data
    if not np.issubdtype(z.values.dtype, np.number):
        raise ValueError("Topic matrix 'z' must contain numerical values")
    
    # Get gene expression matrix
    if isinstance(adata.X, np.ndarray):
        gene_expr = adata.X
    else:
        gene_expr = adata.X.toarray()  # Convert sparse matrix to dense if needed
    
    # Ensure gene expression data is numerical
    if not np.issubdtype(gene_expr.dtype, np.number):
        raise ValueError("Gene expression matrix must contain numerical values")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(gene_expr)) or np.any(np.isinf(gene_expr)):
        raise ValueError("Gene expression matrix contains NaN or infinite values")
    
    if np.any(z.isna()) or np.any(np.isinf(z.values)):
        raise ValueError("Topic matrix contains NaN or infinite values")
    
    # Initialize correlation matrix
    n_genes = gene_expr.shape[1]
    n_topics = z.shape[1]
    correlations = np.zeros((n_genes, n_topics))
    p_values = np.zeros((n_genes, n_topics))
    
    # Compute correlations for each gene-topic pair
    for i in range(n_genes):
        for j in range(n_topics):
            corr, p_val = pearsonr(gene_expr[:, i], z.iloc[:, j].values)
            correlations[i, j] = corr
            p_values[i, j] = p_val
    
    # Create DataFrames with gene names and topic names
    correlation_df = pd.DataFrame(
        correlations,
        index=adata.var_names,
        columns=z.columns if z.columns is not None else [f'Topic_{i}' for i in range(n_topics)]
    )
    
    pvalue_df = pd.DataFrame(
        p_values,
        index=adata.var_names,
        columns=z.columns if z.columns is not None else [f'Topic_{i}' for i in range(n_topics)]
    )
    
    return correlation_df, pvalue_df

    """
    Get top correlated genes for each topic.
    
    Parameters
    ----------
    correlation_df : pandas.DataFrame
        DataFrame containing correlation coefficients
    n_genes : int
        Number of top genes to return per topic
    absolute : bool
        If True, rank by absolute correlation values
        If False, rank by raw correlation values
    
    Returns
    -------
    dict
        Dictionary mapping topic names to top correlated genes
    """
    top_genes = {}
    for topic in correlation_df.columns:
        if absolute:
            # Get absolute correlation values
            abs_corr = correlation_df[topic].abs()
            # Sort and get top n genes
            top_genes[topic] = correlation_df[topic][abs_corr.nlargest(n_genes).index]
        else:
            # Sort by raw correlation values
            top_genes[topic] = correlation_df[topic].nlargest(n_genes)
    return top_genes

def plot_top_positive_correlations_boxplot(adata, z, omics_name, n_top=None, figsize=(12, 6)):
    """
    Create boxplots for each topic showing only the top n positive correlations.
    
    Parameters
    ----------
    correlation_df : pandas.DataFrame
        DataFrame containing correlation coefficients
    n_top : int
        Number of top positive correlations to include per topic
    figsize : tuple
        Figure size (width, height)
    """
    if n_top is None:
        n_top = 5 if omics_name == 'Proteomics' else 10
    corr_df, _ = compute_gene_topic_correlations(adata, z)
    # Get top positive correlations for each topic
    top_correlations_dict = {}
    for topic in corr_df.columns:
        # Get only positive correlations
        positive_corrs = corr_df[topic][corr_df[topic] > 0]
        # Get top n
        top_correlations_dict[topic] = positive_corrs.nlargest(n_top)
    
    # Convert to long format for plotting
    plot_data = []
    for topic, corrs in top_correlations_dict.items():
        for gene, corr in corrs.items():
            plot_data.append({
                'Topic': topic,
                'Gene': gene,
                'Correlation': corr
            })
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create boxplot
    sns.boxplot(
        data=plot_df,
        x='Topic',
        y='Correlation',
        color='skyblue'
    )
    
    # Customize the plot
    plt.title(f'Distribution of Top {n_top} Positive Correlations with {omics_name} per Topic')
    plt.xlabel('Topic')
    plt.ylabel('Pearson Correlation')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

def plot_topic_correlation_ratio_multimodal(data, omics_names, z, k_values=None, figsize=(10, 8)):
    """
    Plot the log2 fold change of mean top-k correlations between modalities for each topic.
    
    Parameters
    ----------
    data_list : list
        List of AnnData objects, one for each modality
    omics_names : list
        List of strings containing names of each modality
    z : pandas.DataFrame
        Topic matrix where rows are cells and columns are topics
    k_values : dict or None
        Dictionary mapping modality names to their k values for top correlations
        If None, defaults to k=20 for RNA and k=5 for others
    figsize : tuple
        Figure size (width, height)
    """
    # Validate inputs
    if len(data) != len(omics_names):
        raise ValueError("Number of datasets must match number of omics names")
    if len(data) != 2:
        raise ValueError("This function currently supports exactly 2 modalities")
        
    # Set default k values if not provided
    if k_values is None:
        k_values = {name: 5 if name in ['Proteomics'] else 10 for name in omics_names}
    
    # Compute correlations for each modality
    modality_corrs = {}
    for data, name in zip(data, omics_names):
        corr_df, _ = compute_gene_topic_correlations(data, z)
        modality_corrs[name] = corr_df
    
    # Calculate mean top-k correlations for each modality
    topic_means = {name: [] for name in omics_names}
    
    for topic in z.columns:
        for name in omics_names:
            # Get top k correlations for this modality
            top_k_corrs = np.sort(modality_corrs[name][topic].values)[-k_values[name]:]
            topic_means[name].append(np.mean(top_k_corrs))
    
    # Calculate log2 fold change
    log2_fold_changes = np.log2(np.array(topic_means[omics_names[0]]) / 
                               np.array(topic_means[omics_names[1]]))
    
    # Create DataFrame with results
    result_df = pd.DataFrame({
        'Topic': z.columns,
        'Log2 Fold Change': log2_fold_changes
    })
    
    # Sort by log2 fold change
    result_df = result_df.sort_values('Log2 Fold Change', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(result_df)), result_df['Log2 Fold Change'])
    
    # Color bars based on which modality has stronger correlation
    for i, bar in enumerate(bars):
        if result_df['Log2 Fold Change'].iloc[i] > 0:
            bar.set_color('skyblue')  # First modality stronger
        else:
            bar.set_color('lightgreen')  # Second modality stronger
    
    # Customize the plot
    plt.title(f'Log2 Fold Change of Top Correlations\n({omics_names[0]} vs {omics_names[1]})')
    plt.xlabel(f'Log2 Fold Change')
    plt.ylabel('Topics')
    
    # Set topic names as y-axis labels
    plt.yticks(range(len(result_df)), result_df['Topic'])
    
    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label=f'Stronger {omics_names[0]} correlation'),
        Patch(facecolor='lightgreen', label=f'Stronger {omics_names[1]} correlation')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return plt.gcf()