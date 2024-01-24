import numpy as np
import pandas as pd
from torchmetrics import ConfusionMatrix, ROC
import torch
import plotly.graph_objects as go
from sklearn.metrics import auc

from ppiformer.utils.bio import BASE_AMINO_ACIDS, BASE_AMINO_ACIDS_GROUPED, class_to_amino_acid
from ppiref.utils.ppipath import path_to_ppi_id


# TODO Move `.compute().cpu().detach().numpy()` logic in all functions to model classes
def plot_confusion_matrix(confmat: ConfusionMatrix, log_scale: bool = False):
    df = pd.DataFrame(
        confmat.compute().cpu().detach().numpy(),
        index=BASE_AMINO_ACIDS,
        columns=BASE_AMINO_ACIDS
    )
    df = df.loc[BASE_AMINO_ACIDS_GROUPED, BASE_AMINO_ACIDS_GROUPED]

    with np.errstate(divide='ignore'):
        z = df.values if not log_scale else np.log(df.values)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            y=df.index,
            x=df.columns,
            colorscale='Aggrnyl'
        )
    )
    fig.update_layout(yaxis = dict(scaleanchor = 'x'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(autosize=False, width=500, height=500)
    
    return fig

def plot_roc_curve(tpr, fpr, threshold, max_points=1000):

    # Downsample to max_points
    if len(tpr) > max_points:
        indices = np.linspace(0, len(tpr) - 1, max_points).astype(int)
        tpr = tpr[indices]
        fpr = fpr[indices]
        threshold = threshold[indices]

    fig = go.Figure()
    # Add ROC curve trace
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             mode='lines',
                             name=f'ROC Curve',
                             line=dict(color='blue', width=2)))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random Guessing',
                             line=dict(color='black', dash='dash')))
    fig.update_layout(autosize=False, width=500, height=500)
    
    return fig

def plot_pr_curve(precision, recall, threshold, max_points=1000):
    fig = go.Figure()
    # Add PR curve trace

    sorted_indices = recall.argsort()
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]

    # Compute the area under the precision-recall curve
    pr_auc = auc(sorted_recall, sorted_precision)

    # Downsample to max_points
    if len(sorted_recall) > max_points:
        indices = np.linspace(0, len(sorted_recall) - 1, max_points).astype(int)
        sorted_recall = sorted_recall[indices]
        sorted_precision = sorted_precision[indices]

        # Add diagonal reference line
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1],
                             mode='lines',
                             name='dash',
                             line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=recall, y=precision,
                             mode='lines',
                             name=f'PR Curve AUC = {pr_auc:.3f}',
                             line=dict(color='blue', width=2)))
    fig.update_layout(autosize=False, width=500, height=500)
    return fig


def hit_rate(y_proba, y_true, k):
    sorted_indices = torch.argsort(y_proba, descending=True)
    top_k_indices = sorted_indices[:k]
    Nhits_k = y_true[top_k_indices].sum().item()
    Npos = y_true.sum().item()
    rate = Nhits_k / Npos if Npos > 0 else 0
    return rate

def success_rate(y_proba, y_true, k):
    sorted_indices = torch.argsort(y_proba, descending=True)
    top_k_indices = sorted_indices[:k]
    Nhits_k = y_true[top_k_indices].sum().item()
    rate = 1 if Nhits_k > 0 else 0
    return rate

def plot_hit_rate_curve(y_proba, y_true, y_case, case2int, ks):   
    rates = []
    cases = list(case2int.values())
    print(cases)
    for k in ks:
        rates.append(0)
        for case in cases:
            y_proba_case = y_proba[y_case == case]
            y_true_case = y_true[y_case == case]
            rate = hit_rate(y_proba_case, y_true_case, k)/len(cases)
            rates[-1] += rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=rates,
                                mode='lines',
                                name=f'Hit Rate Curve',
                                line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1],
                             mode='lines',
                             name='dash',
                             line=dict(color='black', dash='dash')))
    fig.update_layout(autosize=False, width=500, height=500)
    return fig

def plot_success_rate_curve(y_proba, y_true, y_case, case2int, ks):
    rates = []
    cases = list(case2int.values())
    for k in ks:
        rates.append(0)
        for case in cases:
            y_proba_case = y_proba[y_case == case]
            y_true_case = y_true[y_case == case]
            rate = success_rate(y_proba_case, y_true_case, k)/len(cases)
            rates[-1] += rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=rates,
                                mode='lines',
                                name=f'Success Rate Curve',
                                line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1],
                             mode='lines',
                             name='dash',
                             line=dict(color='black', dash='dash')))
    fig.update_layout(autosize=False, width=500, height=500)
    return fig

# NOTE: Works only for single-point masking
def plot_classification_heatmap_example(data, y_proba):
    # Construct dataframe
    nodes = np.hstack(data.node_id)[~data.node_mask.cpu().detach()]
    pdb_ids = [path.split('/')[-1].rsplit('.', 1)[0] for path in data.path]
    idx = [':'.join([pdb, n]) for n, pdb in zip(nodes, pdb_ids)]
    df = pd.DataFrame(
        y_proba.cpu().detach().numpy(),
        columns=BASE_AMINO_ACIDS,
        index=idx
    )
    df = df[BASE_AMINO_ACIDS_GROUPED]

    # Define text
    wts = data.y[~data.node_mask]
    text = np.full((len(wts), 20), '')
    for n, a in zip(np.arange(len(wts)), wts):
        text[n, a] = class_to_amino_acid(a)

    # Plot
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            y=df.index,
            x=df.columns,
            text=text,
            texttemplate='%{text}',
            textfont={'size': 10}
        )
    )
    fig.update_layout(yaxis = dict(scaleanchor = 'x'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(autosize=False, width=500, height=500)

    return fig


def plot_ddg_scatter(ddg_pred, ddg_true, batch):
    ddg_true = ddg_true.cpu().detach().numpy()
    ddg_pred = ddg_pred.cpu().detach().numpy()
    batch = batch.cpu().detach().numpy()

    # TODO pass sample_paths for text=ppi_ids
    # sample_ppi_ids = list(map(path_to_ppi_id, sample_paths))
    # ppi_ids = np.array(sample_ppi_ids)[batch]

    fig = go.Figure(
        data=go.Scatter(
            x=ddg_true,
            y=ddg_pred,
            mode='markers',
            marker_color=batch
        )
    )
    fig.update_layout(
        xaxis_title='True ddG',
        yaxis_title='Predicted ddG'
    )
    return fig
