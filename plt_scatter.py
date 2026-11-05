"""Plot prediction CSV files exported by evaluate.py."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, default=Path('results/evaluation'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/plots'))
    parser.add_argument('--splits', nargs='+', default=['test2016', 'test2013'])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = {'train': '#F9D75C', 'val': '#B18EDC', 'test2016': '#F77A7A', 'test2013': '#73B9E8'}
    for phase in args.splits:
        with (args.predictions_dir / f'{phase}_predictions.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        targets = np.array([float(row['experimental_affinity']) for row in rows])
        predictions = np.array([float(row['predicted_affinity']) for row in rows])
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(targets, predictions, s=8, alpha=0.5, color=colors.get(phase, '#73B9E8'))
        limits = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
        ax.plot(limits, limits, 'r--', lw=2, label='y = x')
        ax.set(xlabel='Experimental affinity', ylabel='Predicted affinity', title=phase)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(args.output_dir / f'{phase}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
