"""Combine the first N values of two single-column CSV files into one file, sequentially."""
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("file1", nargs="?", default="01 - m1_half_shaft_speed_no_mechanical_load.csv")
parser.add_argument("file2", nargs="?", default="03 - m1_mechanically_imbalanced_half_speed.csv")
parser.add_argument("-o", "--output", default="dataset_01_03_combined.csv")
parser.add_argument("-n", "--count", type=int, default=50000, help="number of values to take from each file")
parser.add_argument("--header", default="Combined Vectors", help="header line for the output file")
args = parser.parse_args()


def read_values(path, count):
    with open(path, "r") as f:
        next(f)  # skip header
        values = [next(f).strip() for _ in range(count)]
    return values


def main():
    path1 = os.path.join(SCRIPT_DIR, args.file1)
    path2 = os.path.join(SCRIPT_DIR, args.file2)
    out_path = os.path.join(SCRIPT_DIR, args.output)

    values1 = read_values(path1, args.count)
    values2 = read_values(path2, args.count)

    with open(out_path, "w") as f:
        f.write(args.header + "\n")
        f.write("\n".join(values1 + values2) + "\n")

    print(f"Wrote {len(values1) + len(values2)} values to {out_path}")


if __name__ == "__main__":
    main()
