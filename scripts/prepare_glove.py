"""Convert a GloVe .txt vector file into the CSV format read by symwarp-fastdec.

Drops the word column, min-max scales each feature to [-1, 1], and writes one
point per row with 6 decimal places. The README's expected GloVe centers were
produced with exactly this preprocessing.

Usage:
    python scripts/prepare_glove.py wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt out_dir/
    -> out_dir/glove_<num_points>_<dim>d.csv

Requires numpy and scikit-learn.
"""
import argparse
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_glove(filepath):
    """Return the numeric vectors of a GloVe .txt file, skipping the word column.

    Words may contain spaces, so the dimension is detected from the first line
    and the last `dim` tokens of every line are taken as the vector.
    """
    rows = []
    dim = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if dim is None:
                for i in range(1, len(parts)):
                    try:
                        [float(x) for x in parts[i:]]
                        dim = len(parts) - i
                        break
                    except ValueError:
                        continue
            values = [float(x) for x in parts[-dim:]]
            if len(values) != dim:
                raise ValueError(f"Row {len(rows)+1} has {len(values)} columns, expected {dim}")
            rows.append(values)
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='GloVe vectors .txt file (unzipped)')
    parser.add_argument('out_dir', help='directory for the output CSV')
    args = parser.parse_args()

    data = load_glove(args.input)
    num_records, num_cols = data.shape
    scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"glove_{num_records}_{num_cols}d.csv")
    np.savetxt(csv_path, scaled, delimiter=",", fmt="%.6f")

    print(f"Wrote {csv_path} ({num_records} points, {num_cols} dims)")


if __name__ == '__main__':
    main()
