"""Trace the three feature stages used by the released t-SNE script."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from runtime import choose_device, load_model, make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/mtgpla_best.pt')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--split', default='test2016')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('results/plots'))
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    stage_a, stage_b, stage_c, labels, ids = [], [], [], [], []
    hooks = [model.fc1.register_forward_hook(lambda mod, inputs, output: stage_b.append(inputs[0].detach().cpu().numpy())),
             model.fc2.register_forward_hook(lambda mod, inputs, output: stage_c.append(output.detach().cpu().numpy()))]
    try:
        with torch.no_grad():
            for batch in make_loader(args.data_root, args.split, args.batch_size):
                ids.extend(list(batch.pdbid))
                batch = batch.to(device)
                # Preserve the original Stage A definition: protein embeddings only.
                features = (model.src_emb(batch.protein) + model.pos_emb(batch.protein)).mean(dim=1)
                stage_a.append(features.cpu().numpy())
                labels.append(batch.y.cpu().numpy().reshape(-1))
                model(batch)
    finally:
        for hook in hooks:
            hook.remove()
    arrays = [np.concatenate(stage) for stage in (stage_a, stage_b, stage_c)]
    labels = np.concatenate(labels)
    if len(labels) < 3:
        raise ValueError('t-SNE requires at least three complexes.')
    high = labels > np.median(labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / f'{args.split}_features.npz', stage_a=arrays[0], stage_b=arrays[1],
             stage_c=arrays[2], labels=labels, pdbid=np.asarray(ids))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = ['(a) Protein embedding', '(b) Before MLP', '(c) Second MLP linear layer']
    perplexity = min(30, max(1, len(labels) // 5 - 1))
    for ax, name, features in zip(axes, names, arrays):
        points = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity).fit_transform(features)
        for mask, color, label in ((~high, '#FFC107', 'Low affinity'), (high, '#d62728', 'High affinity')):
            ax.scatter(points[mask, 0], points[mask, 1], c=color, s=8, alpha=0.6, label=label)
        ax.set(title=name, xlabel='t-SNE 1', ylabel='t-SNE 2')
        ax.legend(markerscale=2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / f'{args.split}_tsne_stages.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
