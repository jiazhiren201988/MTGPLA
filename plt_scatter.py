import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from model import DeepTTG

font_path = str( "/gpfs/chencao/renjiazhi/DeepTGIN/simhei.ttf") 
font_manager.fontManager.addfont(font_path)

prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()

plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda")
batch_size = 96
seed = 42

model_path = Path("/gpfs/chencao/renjiazhi/DeepTGIN/result/DeepTTG_2026-03-23 20:44:04.526124_42/best_model.pt")


phase_names = ['train','val','test2016', 'test2013']  

model = DeepTTG().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

data_loaders = {
    phase: DataLoader(
        TestbedDataset(root='data', dataset=phase),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    ) for phase in phase_names
}

def get_predictions(model, loader, device):
    targets = []
    outputs = []
    
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data = data.to(device)
            y_hat, _, _ = model(data)  
            
            targets.append(data.y.cpu().numpy().reshape(-1))
            outputs.append(y_hat.cpu().numpy().reshape(-1))
    
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    return targets, outputs

def plot_and_save(targets, outputs, phase_name):
    plt.figure(figsize=(7, 7))
    color_map = {
    'train': '#F9D75C',      
    'val': '#B18EDC',        
    'test2016': '#F77A7A',   
    'test2013': '#73B9E8'    
    }
    
    plt.scatter(targets, outputs, s=8, alpha=0.5, c=color_map[phase])
    
    min_val = min(targets.min(), outputs.min())
    max_val = max(targets.max(), outputs.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="y = x")
    
    plt.xlabel("Real Affinity")
    plt.ylabel("Predicted Affinity")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = model_path.parent / f"{phase_name}_scatter.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 图片已保存: {save_path}")

if __name__ == "__main__":
    for phase in phase_names:
        print(f"\n正在处理 {phase}...")
        true_vals, pred_vals = get_predictions(model, data_loaders[phase], device)
        plot_and_save(true_vals, pred_vals, phase)

    print("\n🎉 所有散点图生成完成！")
