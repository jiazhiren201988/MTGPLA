import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from model import DeepTTG  

font_path = str( "/gpfs/chencao/linxinyu/projects/DeepTGIN/simhei.ttf")
font_manager.fontManager.addfont(font_path)

prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()

plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda")
batch_size = 285

model_path = Path("/gpfs/chencao/linxinyu/projects/DeepTGIN/result/DeepTTG_2026-03-23 20:44:04.526124_42/best_model.pt")
dataset_name = 'test2016'
output_dir = model_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

model = DeepTTG().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

dataset = TestbedDataset(root='data', dataset=dataset_name)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
print(f"数据集 {dataset_name} 加载完成，共 {len(dataset)} 个样本")

def get_embedding_features(model, data):
    target = data.protein.to(device)
    pocket = data.pocket.to(device)
    smiles = data.smiles.to(device)

    with torch.no_grad():
        prot_emb = model.src_emb(target) + model.pos_emb(target)  
        pocket_emb = model.src_emb(pocket) + model.pos_emb(pocket) 

        smiles_emb = model.CNN_Transformer.emb(smiles.long().to(device)) 

        prot_emb_pool = prot_emb.mean(dim=1)   
        # pocket_emb_pool = pocket_emb.mean(dim=1)  
        # smiles_emb_pool = smiles_emb.mean(dim=1)   

        feat = torch.cat([prot_emb_pool], dim=1) 
    return feat.cpu().numpy()

features_b = [] 
features_c = []  

def hook_fc1_input(module, input, output):
    features_b.append(input[0].detach().cpu().numpy())

def hook_fc2_output(module, input, output):
    features_c.append(output.detach().cpu().numpy())

fc1_handle = model.fc1.register_forward_hook(hook_fc1_input)
fc2_handle = model.fc2.register_forward_hook(hook_fc2_output)

all_labels = []
all_embeddings_a = []  

print("正在提取特征...")
with torch.no_grad():
    for data in tqdm(loader, desc="Processing batches"):
        emb_feat = get_embedding_features(model, data)
        all_embeddings_a.append(emb_feat)
        
        data = data.to(device)
        _ = model(data)  
        
        labels = data.y.cpu().numpy().reshape(-1)
        all_labels.append(labels)

fc1_handle.remove()
fc2_handle.remove()

labels = np.concatenate(all_labels)
features_a = np.concatenate(all_embeddings_a, axis=0)
features_b = np.concatenate(features_b, axis=0)
features_c = np.concatenate(features_c, axis=0)

print(f"特征维度: A={features_a.shape}, B={features_b.shape}, C={features_c.shape}")

median = np.median(labels)
groups = np.where(labels > median, 1, 0) 
colors = np.where(groups == 1, '#d62728', '#1f77b4') 
labels_name = np.where(groups == 1, '高亲和力', '低亲和力')

perplexity = min(30, max(1, len(labels) // 5 - 1))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

X_a = features_a
X_b = features_b
X_c = features_c

tsne_a = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X_a)
tsne_b = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X_b)
tsne_c = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X_c)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
stage_names = ['(a) 嵌入层输出', '(b) 输入MLP之前', '(c) MLP第2层输出']
tsne_results = [tsne_a, tsne_b, tsne_c]

for ax, title, res in zip(axes, stage_names, tsne_results):
    high_mask = groups == 1
    low_mask = groups == 0
    ax.scatter(res[low_mask, 0], res[low_mask, 1], c='#FFC107', s=8, alpha=0.6, label='低亲和力')
    ax.scatter(res[high_mask, 0], res[high_mask, 1], c='#d62728', s=8, alpha=0.6, label='高亲和力')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=2, fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
save_path = output_dir / f'{dataset_name}_tsne_stages.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {save_path}")
plt.show()