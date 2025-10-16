"""
Simple soma classifier using sklearn Pipeline for clean preprocessing + training.


Example usage:
# Train
results = train_soma_classifier_pipeline(
    df_filtered, 
    feat_cols, 
    label_col='pseudo_soma',
    n_splits=10
)

# Save (ONE file!)
save_soma_classifier(
    results, 
    output_path='/root/capsule/results/soma_classifier_v1.joblib',
    metadata={'datasets': datasets, 'date': '2025-10-03'}
)

# Later, load and use:
bundle = load_soma_classifier('/root/capsule/results/soma_classifier_v1.joblib')

df_pred = predict_soma_labels(
    df=df_new,
    pipeline=bundle['pipeline'],
    feat_cols=bundle['feat_cols'],
    threshold=0.5
)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    classification_report, 
    confusion_matrix,
    roc_curve, 
    precision_recall_curve
)
import joblib


# ----
# For psuedolabeling
# ---
def volume_filter(df):
    vol_low = np.percentile(df["volume_um"].values, 10)
    vol_high = np.percentile(df["volume_um"].values, 98)
    df_filtered = df[(df["volume_um"] >= vol_low) & (df["volume_um"] <= vol_high)]
    print(f"Filtered {len(df_filtered)} / {len(df)} cells by volume")

    return df_filtered

def build_features(df: pd.DataFrame):
    out = df.copy()
    eps = 1e-9
    out["log_volume"] = np.log(out["volume_um"].clip(lower=eps))
    out["log_surface_area"] = np.log(out["surface_area"].clip(lower=eps))
    out["sa_per_vol"] = (out["surface_area"] / out["volume_um"]).replace([np.inf, -np.inf], np.nan)

    out.loc[:, "width"] = out["bbox_max_x"] - out["bbox_min_x"]
    out.loc[:, "height"] = out["bbox_max_y"] - out["bbox_min_y"]
    out.loc[:, "depth"] = out["bbox_max_z"] - out["bbox_min_z"]
    out.loc[:, "aspect_ratio"] = out["width"] / out["height"]
    # clean inf in aspect_ratio
    out.loc[:, "aspect_ratio"] = out["aspect_ratio"].replace([np.inf, -np.inf], np.nan)
    out.loc[:, "aspect_ratio"] = out["aspect_ratio"].fillna(0)

    feat_cols = ["log_volume","log_surface_area","sphericity","elongation","solidity","sa_per_vol","aspect_ratio"]
    return out, feat_cols

def make_pca_plots(pcs: np.ndarray, clusters: np.ndarray, proba_soma: np.ndarray, outdir: str):
    outdir = Path(outdir)
    print(f"Saving PCA plots to {outdir}")
    # by cluster
    plt.figure()
    plt.scatter(pcs[:, 0], pcs[:, 1], s=2, c=clusters, alpha=0.3)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (KMeans clusters)")
    plt.tight_layout(); plt.savefig(outdir / "pca_by_cluster_1v2.png", dpi=160); plt.close()

    # plot pc2 v pc3
    plt.figure()
    plt.scatter(pcs[:, 1], pcs[:, 2], s=3, c=clusters, alpha=0.2)
    plt.xlabel("PC2"); plt.ylabel("PC3"); plt.title("PCA (KMeans clusters, PC2 vs PC3)")
    plt.tight_layout(); plt.savefig(outdir / "pca_by_cluster_2v3.png", dpi=160); plt.close()

    # by pseudo-prob
    plt.figure()
    plt.scatter(pcs[:, 0], pcs[:, 1], s=6, c=proba_soma)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA (pseudo-prob soma)")
    plt.tight_layout(); plt.savefig(outdir / "pca_by_pseudo_prob.png", dpi=160); plt.close()


def plot_pca_loadings_bar(
    pca, feature_names, pc=1, top_k=12, scale_by_sdev=True, title=None, out_path=None
):
    """
    Bar plot of the top-|loading| features for one principal component.

    Parameters
    ----------
    pca : fitted sklearn PCA
    feature_names : list[str] of length n_features
    pc : 1-based index of the PC to plot (pc=1 -> PC1)
    top_k : show top_k by absolute contribution
    scale_by_sdev : if True, show *loadings* = components_ * sqrt(explained_variance_)
                    (often what people mean by 'feature weights');
                    if False, show raw components_ (unit-length directions).
    """
    comp_idx = pc - 1
    comp = pca.components_[comp_idx].copy()              # shape (n_features,)
    if scale_by_sdev:
        comp = comp * np.sqrt(pca.explained_variance_[comp_idx])

    # pick top |weights|
    idx = np.argsort(np.abs(comp))[::-1][:top_k]
    vals = comp[idx]
    names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * top_k)))
    ax.barh(range(len(vals)), vals)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Loading" if scale_by_sdev else "Component weight")
    ax.set_title(title or f"PC{pc} feature {'loadings' if scale_by_sdev else 'weights'}")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig, ax


def plot_pca_loadings_heatmap(
    pca, feature_names, n_components=3, scale_by_sdev=True, title="PCA feature loadings", out_path=None
):
    """
    Heatmap of feature loadings for the first n_components PCs.
    Rows = features, Cols = PCs.
    """
    n_components = min(n_components, pca.components_.shape[0])
    mat = pca.components_[:n_components, :].T  # (n_features, n_components)
    if scale_by_sdev:
        mat = mat * np.sqrt(pca.explained_variance_[:n_components])[None, :]

    fig, ax = plt.subplots(figsize=(1.2*n_components + 4, 0.25*len(feature_names) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="bwr")
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)

def pseudo_label_via_dbscan(
    X_scaled,
    df_feats,                          # must include 'sphericity' and 'elongation'
    expected_soma_frac: float = 0.95,  # soma likely the majority
    eps: float | None = None,          # if None, auto from k-distance quantile
    min_samples: int = 20,
    eps_quantile: float = 0.90,        # used when eps=None; try 0.85–0.95
    metric: str = "euclidean",
    greedy_cover: bool = True,         # allow multiple soma-like clusters to hit expected_frac
):
    """
    Density-based pseudo-labeling with DBSCAN.

    Returns
    -------
    labels      : (n,) int      # DBSCAN labels (-1 for noise, 0..C-1)
    pseudo_y    : (n,) int      # 1 = soma, 0 = non-soma
    proba_soma  : (n,) float    # soft probability [0,1] via local density (soma clusters only)
    info        : dict          # debugging metadata (eps, counts, chosen clusters)
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    Xs = np.asarray(X_scaled, dtype=float)
    n = Xs.shape[0]

    # ---- Auto eps via k-distance quantile (distance to the min_samples-th neighbor) ----
    if eps is None:
        k = max(2, int(min_samples))
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(Xs)
        dists = nbrs.kneighbors(Xs, return_distance=True)[0][:, -1]  # kth-NN distance
        eps = float(np.quantile(dists, eps_quantile))
        if not np.isfinite(eps) or eps <= 0:
            eps = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0

    # ---- Fit DBSCAN ----
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(Xs)  # -1 = noise

    uniq = np.unique(labels)
    cluster_ids = [c for c in uniq if c != -1]
    counts = {c: int((labels == c).sum()) for c in cluster_ids}
    n_clusters = len(cluster_ids)

    # Degenerate case: all noise or a single tiny cluster
    if n_clusters == 0:
        pseudo_y = np.zeros(n, dtype=int)
        proba = np.zeros(n, dtype=float)
        return labels, pseudo_y, proba, {"eps": eps, "min_samples": min_samples, "clusters": {}, "chosen": []}

    # ---- Score clusters by shape heuristic: soma ⇒ high sphericity, low elongation ----
    scores = []
    for c in cluster_ids:
        m = labels == c
        med_s = float(np.nanmedian(df_feats.loc[m, "sphericity"]))
        med_e = float(np.nanmedian(df_feats.loc[m, "elongation"]))
        score = med_s - med_e
        frac = counts[c] / max(1, n)
        scores.append((c, score, frac, med_s, med_e))
    # Sort best-first by score
    scores.sort(key=lambda t: t[1], reverse=True)

    # ---- Choose soma cluster(s) ----
    soma_clusters = []
    if greedy_cover:
        target = expected_soma_frac * n
        covered = 0
        for c, s, frac, *_ in scores:
            soma_clusters.append(c)
            covered += counts[c]
            if covered >= target:
                break
        # Guardrail: if total is still tiny (<< expected), fall back to include the largest cluster(s)
        if covered < 0.5 * target:
            soma_clusters = [c for c, *_ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
            covered = 0
            soma_clusters_accum = []
            for c in soma_clusters:
                soma_clusters_accum.append(c)
                covered += counts[c]
                if covered >= target:
                    soma_clusters = soma_clusters_accum
                    break
    else:
        # Single "best" cluster; if it's too small vs expectation, pick the largest instead
        best = scores[0][0]
        if counts[best] / n < expected_soma_frac / 2.0:
            best = max(counts, key=counts.get)
        soma_clusters = [best]

    # ---- Map to pseudo labels ----
    soma_mask = np.isin(labels, soma_clusters)
    pseudo_y = soma_mask.astype(int)

    # ---- Soft probability via local density proxy: 1 / (kNN distance) normalized within soma ----
    # Reuse kNN distances from earlier if available; otherwise recompute a small k
    k_prob = max(5, min_samples // 2)
    nbrs_p = NearestNeighbors(n_neighbors=k_prob, metric=metric).fit(Xs)
    d_k = nbrs_p.kneighbors(Xs, return_distance=True)[0][:, -1]  # kth-NN distance per point
    density = 1.0 / (d_k + 1e-9)

    proba = np.zeros(n, dtype=float)
    if soma_mask.any():
        dens_soma = density[soma_mask]
        dmin, dmax = float(np.min(dens_soma)), float(np.max(dens_soma))
        if dmax > dmin:
            proba[soma_mask] = (dens_soma - dmin) / (dmax - dmin)
        else:
            proba[soma_mask] = 1.0  # uniform density inside soma clusters
    # non-soma (incl. noise) remain 0

    info = {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "counts": counts,
        "scores": scores,            # (cluster_id, score, frac, med_s, med_e)
        "chosen": soma_clusters,
    }
    return labels, pseudo_y, proba, info

def do_pseudolabel_clustering(df, feat_cols):

    print(f"Running PCA and clustering on {df.shape[0]} cells")
    print(f"Using features: {feat_cols}")
    X_raw = df[feat_cols].to_numpy()
    object_ids = df["cell_id"].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_raw)
    X = scaler.fit_transform(X_imp)

    n_pca = max(2, 7)
    pca = PCA(n_components=n_pca, random_state=42)
    pcs = pca.fit_transform(X)
    print(pca.explained_variance_ratio_.tolist())

    pcs = pca.transform(X)  # already scaled
    plot_pca_loadings_bar(pca, feat_cols, pc=1, top_k=10, out_path="pc1_loadings.png")
    plot_pca_loadings_heatmap(pca, feat_cols, n_components=3, out_path="pca_loadings_heatmap.png")


    Z_for_cluster = pcs[:, :2] if True else X

    # method 1
    # kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    # clusters = kmeans.fit_predict(Z_for_cluster)
    # dists = kmeans.transform(Z_for_cluster)
    # soft = soft_assign_from_dist(dists)
    # soma_cluster = choose_soma_cluster(df, clusters)
    # pseudo_y = (clusters == soma_cluster).astype(int)
    # proba_soma = soft[:, soma_cluster]

    # method 2
    # clusters, pseudo_y, proba_soma, Z = pseudo_label_via_spectral_kmeans(X, df, expected_soma_frac=0.8,
    #                                                                      n_components=12, gamma=1)

    # method 3
    #clusters, pseudo_y, proba_soma = pseudo_label_via_hierarchical_aggressive(X, df,cut_quantile=.15)

    clusters, pseudo_y, proba_soma, info = pseudo_label_via_dbscan(Z_for_cluster, 
                                                                   df, expected_soma_frac=0.8,
                                                                   min_samples=50,
                                                                   eps=None, # default none
                                                                   eps_quantile=0.95) # 0.9 default
    
    # add clusters to df_filtered
    df["cluster"] = clusters
    # set pseudo labels, True is cluster col = 0
    df["pseudo_soma"] = df["cluster"] == 0
    df["proba_soma"] = proba_soma
    # PLOTS
    base_dir = Path("/root/capsule/scratch/seg_metrics_cluster")
    base_dir.mkdir(exist_ok=True, parents=True)
    make_pca_plots(pcs, df["pseudo_soma"].values, proba_soma, base_dir)


    return pcs, df, info

def plot_pca_3d(pcs, color=None, title="PCA (3D)", out_path=None,
                figsize=(7, 5), elev=18, azim=35, point_size=6):
    """
    Plot a 3D PCA scatter.

    Parameters
    ----------
    pcs : np.ndarray, shape (n_samples, >=3)
        PCA scores (e.g., from sklearn.decomposition.PCA).
    color : array-like or None
        Optional values to color points (cluster ids or continuous scores).
    title : str
        Figure title.
    out_path : str or None
        If provided, save the figure to this path (e.g., "pca_3d.png").
    figsize : tuple
        Matplotlib figsize.
    elev, azim : float
        3D view angles.
    point_size : int
        Marker size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (ensures 3D proj is registered)

    assert pcs.shape[1] >= 3, "pcs must have at least 3 components"

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    kwargs = {"s": point_size}
    if color is not None:
        kwargs["c"] = color

    sc = ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], **kwargs, cmap="viridis")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    # # add legend for cluster labels
    # plt.legend(*sc.legend_elements(), title="Clusters")

    # add colorbar
    if color is not None:
        fig.colorbar(sc, ax=ax, shrink=0.7)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160)
    return fig, ax


def choose_soma_cluster(df_feats: pd.DataFrame, clusters: np.ndarray) -> int:
    scores = []
    for c in np.unique(clusters):
        med_s = np.nanmedian(df_feats.loc[clusters == c, "sphericity"])
        med_e = np.nanmedian(df_feats.loc[clusters == c, "elongation"])
        score = med_s - med_e
        scores.append((c, score))
    scores.sort(key=lambda t: t[1], reverse=True)
    return int(scores[0][0])

# ----
# Train classifier
# ----
def train_soma_classifier_pipeline(
    df: pd.DataFrame,
    feat_cols: List[str],
    label_col: str = 'pseudo_soma',
    n_splits: int = 10,
    random_state: int = 42,
    class_weight: str = 'balanced',
    figsize: Tuple[int, int] = (12, 5)
) -> Dict:
    """
    Train logistic regression soma classifier using sklearn Pipeline.
    
    Pipeline includes:
    1. SimpleImputer (median strategy)
    2. StandardScaler
    3. LogisticRegression (with class balancing)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and labels
    feat_cols : list
        List of feature column names
    label_col : str
        Column name for binary labels (default 'pseudo_soma')
    n_splits : int
        Number of CV folds (default 10)
    random_state : int
        Random seed
    class_weight : str or dict
        'balanced' automatically adjusts weights inversely proportional to class frequencies
    figsize : tuple
        Figure size for plots
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'pipeline': Fitted sklearn Pipeline
        - 'feat_cols': Feature column names (in order!)
        - 'cv_metrics': Cross-validation metrics
        - 'feature_importance': DataFrame with coefficients
    """
    
    # Prepare data
    X_raw = df[feat_cols].to_numpy()
    y = df[label_col].to_numpy()
    
    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        ))
    ])
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Storage for CV results
    fold_metrics = {
        'roc_auc': [],
        'avg_precision': [],
        'accuracy': [],
        'balanced_accuracy': []
    }
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    print(f"Training pipeline with {n_splits}-fold CV")
    print(f"Class distribution: {np.bincount(y)} (0: non-soma, 1: soma)")
    print(f"Class balance: {y.mean():.2%} soma\n")
    
    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train, X_val = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone and train pipeline
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = fold_pipeline.predict(X_val)
        y_proba = fold_pipeline.predict_proba(X_val)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_proba)
        avg_prec = average_precision_score(y_val, y_proba)
        acc = (y_pred == y_val).mean()
        
        # Balanced accuracy
        tn = ((y_pred == 0) & (y_val == 0)).sum()
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        
        fold_metrics['roc_auc'].append(roc_auc)
        fold_metrics['avg_precision'].append(avg_prec)
        fold_metrics['accuracy'].append(acc)
        fold_metrics['balanced_accuracy'].append(balanced_acc)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        print(f"Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, AP={avg_prec:.4f}, "
              f"Acc={acc:.4f}, Bal-Acc={balanced_acc:.4f}")
    
    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    print("\n" + "="*60)
    print("Cross-Validation Results (mean ± std):")
    print("="*60)
    for metric_name, values in fold_metrics.items():
        print(f"{metric_name:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    print("\nAggregated Confusion Matrix:")
    print(cm)
    print(f"\nTrue Negatives:  {cm[0,0]:6d}  |  False Positives: {cm[0,1]:6d}")
    print(f"False Negatives: {cm[1,0]:6d}  |  True Positives:  {cm[1,1]:6d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, 
                                target_names=['Non-soma', 'Soma'], 
                                digits=4))
    
    # Train final pipeline on ALL data
    print("\nTraining final pipeline on full dataset...")
    pipeline.fit(X_raw, y)
    
    # Plot ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    roc_auc_final = roc_auc_score(all_y_true, all_y_proba)
    axes[0].plot(fpr, tpr, linewidth=2, 
                 label=f'ROC (AUC = {roc_auc_final:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve (10-Fold CV)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_proba)
    ap_final = average_precision_score(all_y_true, all_y_proba)
    axes[1].plot(recall, precision, linewidth=2,
                 label=f'PR (AP = {ap_final:.3f})')
    baseline = all_y_true.mean()  # prevalence
    axes[1].axhline(baseline, color='k', linestyle='--', linewidth=1,
                    label=f'Baseline ({baseline:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (10-Fold CV)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance (from final model)
    classifier = pipeline.named_steps['classifier']
    feature_importance = pd.DataFrame({
        'feature': feat_cols,
        'coefficient': classifier.coef_[0],
        'abs_coefficient': np.abs(classifier.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nFeature Importance (by absolute coefficient):")
    print(feature_importance.to_string(index=False))
    
    # Return results
    results = {
        'pipeline': pipeline,
        'feat_cols': feat_cols,
        'fold_metrics': fold_metrics,
        'mean_metrics': {k: np.mean(v) for k, v in fold_metrics.items()},
        'std_metrics': {k: np.std(v) for k, v in fold_metrics.items()},
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'y_proba': all_y_proba,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }
    
    return results


def predict_soma_labels(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feat_cols: List[str],
    threshold: float = 0.5,
    prob_col: str = 'soma_probability',
    label_col: str = 'predicted_soma'
) -> pd.DataFrame:
    """
    Apply trained pipeline to new data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns
    pipeline : Pipeline
        Trained sklearn Pipeline
    feat_cols : list
        Feature column names (must match training order!)
    threshold : float
        Probability threshold for binary classification
    prob_col : str
        Name for probability column
    label_col : str
        Name for binary label column
        
    Returns
    -------
    df_pred : pd.DataFrame
        Copy of input with added prediction columns
    """
    # Validate features
    missing_cols = [col for col in feat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")
    
    df_pred = df.copy()
    
    # Extract features (raw, pipeline handles preprocessing)
    X_raw = df_pred[feat_cols].to_numpy()
    
    # Predict (pipeline does imputation + scaling automatically)
    proba = pipeline.predict_proba(X_raw)[:, 1]
    labels = (proba >= threshold).astype(int)
    
    # Add to dataframe
    df_pred[prob_col] = proba
    df_pred[label_col] = labels
    
    # Print summary
    n_soma = labels.sum()
    n_total = len(labels)
    print(f"Inference complete on {n_total} samples")
    print(f"Predicted soma: {n_soma} ({100*n_soma/n_total:.2f}%)")
    print(f"Predicted non-soma: {n_total - n_soma} ({100*(n_total-n_soma)/n_total:.2f}%)")
    print(f"\nProbability statistics:")
    print(f"  Mean: {proba.mean():.4f}")
    print(f"  Median: {np.median(proba):.4f}")
    print(f"  Min: {proba.min():.4f}")
    print(f"  Max: {proba.max():.4f}")
    
    return df_pred


def save_soma_classifier(
    results: Dict,
    output_path: Path,
    metadata: Optional[Dict] = None
):
    """
    Save pipeline + metadata to a single file.
    
    Parameters
    ----------
    results : dict
        Output from train_soma_classifier_pipeline()
    output_path : Path
        Path to save .joblib file (e.g., 'soma_classifier.joblib')
    metadata : dict, optional
        Additional metadata to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Bundle everything
    bundle = {
        'pipeline': results['pipeline'],
        'feat_cols': results['feat_cols'],
        'cv_metrics': {
            'mean': results['mean_metrics'],
            'std': results['std_metrics']
        },
        'feature_importance': results['feature_importance'].to_dict('records'),
        'metadata': metadata or {}
    }
    
    joblib.dump(bundle, output_path)
    print(f"✓ Saved classifier bundle to {output_path}")
    print(f"  Pipeline: {len(results['feat_cols'])} features")
    print(f"  ROC-AUC: {results['mean_metrics']['roc_auc']:.4f} ± {results['std_metrics']['roc_auc']:.4f}")


def load_soma_classifier(model_path: Path = None) -> Dict:
    """
    Load saved classifier bundle.

    Parameters
    ----------
    model_path : Path or str
        Path to .joblib file

    Returns
    -------
    bundle : dict
        Dictionary with 'pipeline', 'feat_cols', 'cv_metrics', etc.
    """
    import os
    from pathlib import Path
    import joblib

    # Default to file in same directory as this script if not provided
    if model_path is None:
        model_path = Path(__file__).parent / "simple_soma_1.joblib"
    else:
        model_path = Path(model_path)

    bundle = joblib.load(model_path)

    print(f"✓ Loaded classifier from {model_path}")
    print(f"  Features ({len(bundle['feat_cols'])}): {bundle['feat_cols']}")
    if 'cv_metrics' in bundle:
        print(f"  ROC-AUC: {bundle['cv_metrics']['mean']['roc_auc']:.4f}")

    return bundle


def analyze_threshold_sensitivity(
    df: pd.DataFrame,
    pipeline: Pipeline,
    feat_cols: List[str],
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> np.ndarray:
    """
    Show how predicted soma counts change with different thresholds.
    """
    X_raw = df[feat_cols].to_numpy()
    proba = pipeline.predict_proba(X_raw)[:, 1]
    
    print("Threshold Sensitivity Analysis")
    print("="*50)
    for thresh in thresholds:
        labels = (proba >= thresh).astype(int)
        n_soma = labels.sum()
        pct = 100 * n_soma / len(labels)
        print(f"Threshold {thresh:.2f}: {n_soma:6d} soma ({pct:5.2f}%)")
    
    return proba


# Example usage in notebook:
