"""Evaluate a checkpoint and export per-complex predictions."""
import argparse
import csv
import json
from pathlib import Path

from runtime import choose_device, load_model, make_loader, predict, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mtgpla_best.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", nargs="+", default=["test2016", "test2013"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for phase in args.splits:
        loader = make_loader(args.data_root, phase, args.batch_size)
        ids, targets, predictions = predict(model, loader, device)
        result[phase] = {"n": len(ids), **scores(targets, predictions)}
        with (output / f"{phase}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pdbid", "experimental_affinity", "predicted_affinity"])
            writer.writerows(zip(ids, targets.tolist(), predictions.tolist()))
        print(phase, json.dumps(result[phase]))
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
