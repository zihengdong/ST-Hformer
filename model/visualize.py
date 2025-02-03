import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import sys
import os
import argparse
from datetime import datetime
import pandas as pd

sys.path.append("..")
from model.GRU_Transformer import GRU_Transformer
from lib.data_prepare import get_dataloaders_from_index_data
import yaml

def load_model_and_data(model_path, config_path, dataset="PEMS08", device="cuda"):
    """加载模型和数据"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)[dataset]
    
    model = GRU_Transformer(**cfg["model_args"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    data_path = f"../data/{dataset}"
    trainset_loader, valset_loader, testset_loader, scaler = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64)
    )
    
    return model, testset_loader, scaler, cfg

def visualize_spatial_embedding(model, save_path=None):
    """使用t-SNE可视化空间嵌入"""
    embeddings = model.adaptive_embedding.detach().cpu().numpy()
    # 重塑为(timesteps * nodes, embedding_dim)
    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    # 应用t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization of Spatial Embeddings')
    if save_path:
        plt.savefig(os.path.join(save_path, 'spatial_embedding_tsne.png'))
    plt.close()

def visualize_temporal_correlation(model, save_path=None):
    """可视化时间相关性热图"""
    embeddings = model.adaptive_embedding.detach().cpu().numpy()
    # 计算时间步之间的相关系数
    corr_matrix = np.corrcoef(embeddings.mean(axis=1))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0)
    plt.title('Temporal Correlation Heatmap')
    if save_path:
        plt.savefig(os.path.join(save_path, 'temporal_correlation.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, required=True, help='Path to saved model')
    parser.add_argument('--c', type=str, required=True, help='Path to config file')
    parser.add_argument('--d', type=str, default='PEMS08', help='Dataset name')
    parser.add_argument('--o', type=str, default='../visualization_results', help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(args.o, f'{args.d}_{timestamp}')
    os.makedirs(save_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, testset_loader, scaler, cfg = load_model_and_data(
        args.m, 
        args.c,
        args.d,
        device
    )
    
    print("Generating spatial embedding visualization...")
    visualize_spatial_embedding(model, save_path)
    
    print("Generating temporal correlation visualization...")
    visualize_temporal_correlation(model, save_path)
    
    
    print(f"Visualizations saved to {save_path}")

if __name__ == '__main__':
    main()