import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.GRU_Transformer import GRU_Transformer  
from model.Transformer_module import Transformer_module 
from model.GRU_module import GRU_module 
import yaml
from lib.utils import StandardScaler
from lib.data_prepare import generate_fourier_features


def load_pems08_data(data_dir):
    """
    加载 PEMS08 数据集并提取交通流量数据。

    Args:
        data_dir: PEMS08 数据集的目录.

    Returns:
        np.ndarray: 包含流量数据的NumPy数组，形状为 (时间步数，节点数，特征维度)
    """
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    return data[..., 0]


def load_model(model_path, config_path):
    """
    加载预训练模型
    
    Args:
        model_path: 模型权重文件路径
        config_path: 模型配置文件路径
    """

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['PEMS08']  

    model = GRU_Transformer(**cfg['model_args'])
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, cfg


def predict_sequence(model, data, scaler, cfg, start_idx, seq_len=600):
    """
    预测指定长度的序列
    
    Args:
        model: 加载的模型
        data: 输入数据
        scaler: 数据标准化器
        cfg: 模型配置
        start_idx: 起始索引
        seq_len: 预测序列长度
    """
    predictions = []
    current_idx = start_idx
    
    while len(predictions) < seq_len:

        x = data[current_idx:current_idx + cfg['model_args']['in_steps']]
        x_traffic = x[..., 0:1]
        x_traffic = scaler.transform(x_traffic)
        
        if cfg.get('time_of_day', True):
            tod = x[..., 1:2]
            fourier_features = generate_fourier_features(tod, cfg['model_args']['num_harmonics'])
            x = np.concatenate([x_traffic, tod, x[..., 2:3], fourier_features], axis=-1)
        
        x = torch.FloatTensor(x).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x)
            pred = pred.squeeze(0) 
            pred = scaler.inverse_transform(pred.numpy())
        
     
        predictions.extend(pred[:, :, 0].tolist())
        current_idx += cfg['model_args']['out_steps']
        
        if len(predictions) > seq_len:
            predictions = predictions[:seq_len]
    
    return np.array(predictions)


def visualize_traffic_flow(data, predictions, start_step=0, end_step=600, node_index=0):
    """
    可视化特定节点的真实交通流量和预测值。

    Args:
        data: 真实数据
        predictions: 预测数据
        start_step: 起始时间步长
        end_step: 结束时间步长
        node_index: 需要展示的节点序号（从0开始）
    """
    if end_step > data.shape[0]:
        end_step = data.shape[0]

    traffic_flow = data[start_step:end_step, node_index]
    pred_flow = predictions[start_step:end_step, node_index]

    time_steps = np.arange(start_step, end_step)

    plt.figure(figsize=(15, 12))
    
    plt.plot(time_steps, traffic_flow, label='Ground Truth', color='blue')
    plt.plot(time_steps, pred_flow, label='Prediction', color='red', linestyle='--')
    
    plt.title(f"Traffic Flow Prediction vs Ground Truth for Node {node_index + 1}")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Flow")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'node_{node_index}_prediction.png')
    plt.show()


def visualize_multiple_nodes(data, predictions, start_step=0, end_step=600, nodes=[0,1,2,3,4]):
    """
    可视化多个节点的真实交通流量和预测值。

    Args:
        data: 真实数据
        predictions: 预测数据
        start_step: 起始时间步长
        end_step: 结束时间步长
        nodes: 需要展示的节点列表
    """
    if end_step > data.shape[0]:
        end_step = data.shape[0]


    fig, axes = plt.subplots(len(nodes), 1, figsize=(15, 5*len(nodes)))
    time_steps = np.arange(start_step, end_step)

    for idx, node_index in enumerate(nodes):

        traffic_flow = data[start_step:end_step, node_index, 0]  
        pred_flow = predictions[start_step:end_step, node_index]

        ax = axes[idx]
        ax.plot(time_steps, traffic_flow, label='Ground Truth', color='blue')
        ax.plot(time_steps, pred_flow, label='Prediction', color='red', linestyle='--')
        
       # ax.set_title(f"Traffic Flow for Node {node_index + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Traffic Flow")
        ax.grid(True)
        ax.legend()
        #ax.set_title(f"Traffic Flow for Node {node_index + 1}")
    plt.tight_layout()
    plt.savefig('multiple_nodes_prediction.png')
    plt.show()


if __name__ == "__main__":
    # set up
    data_dir = "/home/liuyq/data/jiang/FFXT15/data/PEMS08"
    model_path = "/home/liuyq/data/jiang/FFXT15/saved_models/model_fold3_20250203_103125.pt"
    config_path = "/home/liuyq/data/jiang/FFXT15/model/GRU_Transformer.yaml"
    
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    model, cfg = load_model(model_path, config_path)
    scaler = StandardScaler(mean=data[..., 0].mean(), std=data[..., 0].std())
    predictions = predict_sequence(model, data, scaler, cfg, start_idx=0, seq_len=600)  
    visualize_multiple_nodes(data, predictions, 0, 600, nodes=[0,1,2,3])