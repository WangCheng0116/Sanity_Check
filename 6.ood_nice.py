"""
4 × 4 benign × malicious OOD experiment (linear SVM)

Given a model directory containing exactly four `benign_*.npy`
and four `malicious_*.npy` hidden-state files, the script

1. builds a 4 × 4 in-distribution (ID) accuracy matrix  
2. builds a 4 × 4 out-of-distribution (OOD) accuracy matrix  
3. saves the results as CSV files and a pair of annotated heat-maps
"""

import argparse
import glob
import os
import random
import urllib.request
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# ---------------------------------------------------------------------
# font setup ─ Times New Roman
# ---------------------------------------------------------------------
FONT_URL = (
    "https://github.com/justrajdeep/fonts/raw/master/Times%20New%20Roman.ttf"
)
FONT_NAME = "Times New Roman"
FONT_PATH = os.path.join(os.path.dirname(__file__), "TimesNewRoman.ttf")

if not os.path.exists(FONT_PATH):
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)

fm.fontManager.addfont(FONT_PATH)
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = FONT_NAME

# Seaborn global aesthetics
sns.set_theme(
    context="notebook",
    style="white",
    font=FONT_NAME,
    rc={"axes.titleweight": "bold", "axes.titlesize": 14},
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_npy_files(model_dir: str) -> Tuple[List[str], List[str]]:
    npy_files = glob.glob(os.path.join(model_dir, "*.npy"))
    benign = sorted(f for f in npy_files if os.path.basename(f).startswith("benign_"))
    malicious = sorted(
        f for f in npy_files if os.path.basename(f).startswith("malicious_")
    )
    if len(benign) != 4 or len(malicious) != 4:
        raise ValueError(
            f"Expect 4+4 files, found {len(benign)} benign and {len(malicious)} malicious"
        )
    return benign, malicious


def _balanced_stack(
    benign_files: List[str], malicious_files: List[str], seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_b = np.vstack([np.load(f) for f in benign_files])
    x_m = np.vstack([np.load(f) for f in malicious_files])
    n = min(len(x_b), len(x_m))
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [rng.choice(x_b, n, replace=False), rng.choice(x_m, n, replace=False)]
    )
    y = np.hstack([np.zeros(n), np.ones(n)])
    return x, y


def train_svm(x: np.ndarray, y: np.ndarray, seed: int) -> SVC:
    clf = SVC(kernel="linear", random_state=seed)
    clf.fit(x, y)
    return clf


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser("4×4 OOD probe (SVM)")
    parser.add_argument("--model_dir", required=True, help="directory with *.npy files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seeds(args.seed)
    benign_files, malicious_files = load_npy_files(args.model_dir)
    b_labels = [
        os.path.basename(f).replace("benign_", "").replace(".npy", "")
        for f in benign_files
    ]
    m_labels = [
        os.path.basename(f).replace("malicious_", "").replace(".npy", "")
        for f in malicious_files
    ]

    id_acc = np.zeros((4, 4))
    ood_acc = np.zeros_like(id_acc)

    for i_b, b_file in enumerate(benign_files):
        for j_m, m_file in enumerate(malicious_files):
            # in-distribution
            x_id, y_id = _balanced_stack([b_file], [m_file], seed=args.seed)
            x_tr, x_te, y_tr, y_te = train_test_split(
                x_id,
                y_id,
                test_size=0.2,
                stratify=y_id,
                random_state=args.seed,
            )
            id_acc[i_b, j_m] = train_svm(x_tr, y_tr, args.seed).score(x_te, y_te)

            # out-of-distribution
            others_b = [f for f in benign_files if f != b_file]
            others_m = [f for f in malicious_files if f != m_file]
            x_res, y_res = _balanced_stack(others_b, others_m, seed=args.seed)
            x_res_tr, _, y_res_tr, _ = train_test_split(
                x_res,
                y_res,
                test_size=0.2,
                stratify=y_res,
                random_state=args.seed,
            )
            clf_ood = train_svm(x_res_tr, y_res_tr, args.seed)
            x_test_ood, y_test_ood = _balanced_stack([b_file], [m_file], args.seed)
            ood_acc[i_b, j_m] = clf_ood.score(x_test_ood, y_test_ood)

    id_df = pd.DataFrame(id_acc, index=b_labels, columns=m_labels)
    ood_df = pd.DataFrame(ood_acc, index=b_labels, columns=m_labels)

    id_df.to_csv(os.path.join(args.model_dir, "id_results.csv"))
    ood_df.to_csv(os.path.join(args.model_dir, "ood_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # left: ID
    sns.heatmap(
        id_df,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        ax=axes[0],
        cbar_kws={"label": "Accuracy"},
        annot_kws={"fontsize": 8},
        vmin=0.5,
        vmax=1.0,
    )
    axes[0].set_title("In-distribution (80 / 20 split)")
    axes[0].set_ylabel("Benign dataset")
    axes[0].set_xlabel("Malicious dataset")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)

    # right: OOD
    sns.heatmap(
        ood_df,
        cmap="Reds",
        annot=True,
        fmt=".3f",
        ax=axes[1],
        cbar_kws={"label": "Accuracy"},
        annot_kws={"fontsize": 8},
        vmin=0.5,
        vmax=1.0,
    )
    axes[1].set_title("Out-of-distribution")
    axes[1].set_ylabel("")            # hide benign-axis label
    axes[1].set_yticklabels([])       # hide benign dataset names
    axes[1].set_xlabel("Malicious dataset")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    out_png = os.path.join(args.model_dir, "heatmaps.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[✓] Saved heat-maps → {out_png}")


if __name__ == "__main__":
    main()