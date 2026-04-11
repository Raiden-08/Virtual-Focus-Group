import os
import torch
import numpy as np

def load_dataset(cfg, seed, mock=False):
    if mock:
        from curriculum.trainer import MockTemporalGraphDataset
        return MockTemporalGraphDataset(), MockTemporalGraphDataset(), MockTemporalGraphDataset()
    from data.swat import load_swat
    from data.smap import load_smap
    dataset_name = cfg['data']['dataset']
    data_dir = cfg['data']['data_dir']
    window = cfg['model']['window_size']
    if dataset_name == 'swat':
        return load_swat(data_dir=data_dir, window=window)
    elif dataset_name == 'smap':
        return load_smap(data_dir=data_dir, window=window)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def load_backbone(cfg, device="cuda", mock=False):
    if mock:
        from curriculum.trainer import MockBackbone
        return MockBackbone()
    from backbone.backbone import Backbone
    model = Backbone(
        d_in=cfg['model']['d_in'],
        hidden_size=cfg['model']['lstm_hidden'],
        gnn_out_dim=cfg['model']['gnn_out_dim'],
        num_nodes=cfg['model']['num_nodes'],
        window_size=cfg['model']['window_size'],
        lstm_layers=cfg['model']['lstm_layers'],
        gat_heads=cfg['model']['gnn_heads'],
        dropout=cfg['model']['dropout']
    )
    return model.to(device)

@torch.no_grad()
def evaluate_on_test(model, test_ds, device="cuda", batch_size=512):
    model.eval()
    all_scores, all_labels = [], []
    
    class DPGraphWrapper:
        def __init__(self, data):
            self.edge_index = data.edge_index
            self.edge_attr = getattr(data, 'edge_attr', None)
            
    n_test = len(test_ds)
    for i in range(0, n_test, batch_size):
        end_i = min(i + batch_size, n_test)
        batch_data = [test_ds[j] for j in range(i, end_i)]
        
        x = torch.stack([d["x"] for d in batch_data]).to(device)
        y = torch.stack([d["y"] for d in batch_data])
        graph_safe = DPGraphWrapper(batch_data[0]["graph"])
        
        # ⚡ UNPACK ALL 3 VALUES (Prevents the crash)
        _, x_hat_all, x_pred_all = model(x, graph_safe)
        
        target_recon = torch.stack([
            torch.tensor(test_ds.signals[d["t"]], dtype=torch.float32) 
            for d in batch_data
        ]).unsqueeze(-1).to(device)

        target_pred = torch.stack([
            torch.tensor(test_ds.signals[min(d["t"] + 1, len(test_ds.signals)-1)], dtype=torch.float32) 
            for d in batch_data
        ]).unsqueeze(-1).to(device)
        
        recon_err = torch.norm(x_hat_all.view(x_hat_all.shape[0], x_hat_all.shape[1], -1) - target_recon.view(target_recon.shape[0], target_recon.shape[1], -1), dim=-1)
        pred_err = torch.norm(x_pred_all.view(x_pred_all.shape[0], x_pred_all.shape[1], -1) - target_pred.view(target_pred.shape[0], target_pred.shape[1], -1), dim=-1)
        
        node_scores = recon_err + pred_err
        system_scores = node_scores.mean(dim=1)
        
        all_scores.extend(system_scores.cpu().tolist())
        all_labels.extend(y[:, 0].tolist())
        
    return all_scores, all_labels