import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

dataset_dir = "runs/exp3/train"


def normalize_contacts(num_contacts, n_elements=100):
    return np.clip(np.log1p(num_contacts) / np.log1p(n_elements), 0.0, 1.0)


def normalize_angular_span(angular_span):
    return np.where(
        angular_span <= 180.0,
        0.8 * angular_span / 180.0,
        0.8 + 0.2 * np.clip((angular_span - 180.0) / 180.0, 0.0, 1.0),
    )


def load_dataset_metrics(dataset_dir):
    files = glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True)

    print(f"Looking in: {os.path.abspath(dataset_dir)}")
    print(f"Found {len(files)} npz files")

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    contact_list = []
    score_list = []
    angular_span_list = []

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            num_contacts = float(np.asarray(data.get("num_contacts", [0.0])).reshape(-1)[0])
            disturbance_score = float(np.asarray(data.get("disturbance_resistance_score", [0.0])).reshape(-1)[0])
            angular_span = float(np.asarray(data.get("angular_span", [0.0])).reshape(-1)[0])

            contact_list.append(num_contacts)
            score_list.append(disturbance_score)
            angular_span_list.append(angular_span)

    contact_arr = np.array(contact_list)
    score_arr = np.array(score_list)
    angular_span_arr = np.array(angular_span_list)

    contact_norm_arr = normalize_contacts(contact_arr, n_elements=100)
    score_norm_arr = np.clip(score_arr, 0.0, 1.0)
    angular_span_norm_arr = normalize_angular_span(angular_span_arr)

    return (
        contact_arr,
        score_arr,
        angular_span_arr,
        contact_norm_arr,
        score_norm_arr,
        angular_span_norm_arr,
    )


def print_one_summary(name, arr):
    print(f"\n=== {name} SUMMARY ===")
    print(f"min:  {arr.min():.3f}")
    print(f"max:  {arr.max():.3f}")
    print(f"mean: {arr.mean():.3f}")
    print(f"std:  {arr.std():.3f}")


def print_summary(contact_arr, score_arr, angular_span_arr,
                  contact_norm_arr, score_norm_arr, angular_span_norm_arr):
    print_one_summary("RAW CONTACT COUNT", contact_arr)
    print_one_summary("RAW DISTURBANCE SCORE", score_arr)
    print_one_summary("RAW ANGULAR SPAN", angular_span_arr)

    print_one_summary("NORMALIZED CONTACT COUNT", contact_norm_arr)
    print_one_summary("NORMALIZED DISTURBANCE SCORE", score_norm_arr)
    print_one_summary("NORMALIZED ANGULAR SPAN", angular_span_norm_arr)

    print(f"\nTotal samples: {len(score_arr)}")


def save_hist(arr, xlabel, title, save_path, bins=20):
    plt.figure()
    plt.hist(arr, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_distributions(contact_arr, score_arr, angular_span_arr,
                       contact_norm_arr, score_norm_arr, angular_span_norm_arr,
                       dataset_dir):
    plot_dir = os.path.join(dataset_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # -------------------------
    # Raw values
    # -------------------------
    save_hist(
        contact_arr,
        "Number of Contacts",
        "Raw Contact Count Distribution",
        os.path.join(plot_dir, "raw_contact_distribution.png"),
    )

    save_hist(
        score_arr,
        "Disturbance Resistance Score",
        "Raw Disturbance Score Distribution",
        os.path.join(plot_dir, "raw_disturbance_score_distribution.png"),
    )

    save_hist(
        angular_span_arr,
        "Angular Span (deg)",
        "Raw Angular Span Distribution",
        os.path.join(plot_dir, "raw_angular_span_distribution.png"),
    )

    # -------------------------
    # Normalized values
    # -------------------------
    save_hist(
        contact_norm_arr,
        "Normalized Contact Count",
        "Normalized Contact Count Distribution",
        os.path.join(plot_dir, "norm_contact_distribution.png"),
    )

    save_hist(
        score_norm_arr,
        "Normalized Disturbance Score",
        "Normalized Disturbance Score Distribution",
        os.path.join(plot_dir, "norm_disturbance_score_distribution.png"),
    )

    save_hist(
        angular_span_norm_arr,
        "Normalized Angular Span",
        "Normalized Angular Span Distribution",
        os.path.join(plot_dir, "norm_angular_span_distribution.png"),
    )

    print(f"\nSaved plots to: {plot_dir}")


if __name__ == "__main__":
    (
        contact_arr,
        score_arr,
        angular_span_arr,
        contact_norm_arr,
        score_norm_arr,
        angular_span_norm_arr,
    ) = load_dataset_metrics(dataset_dir)

    print_summary(
        contact_arr,
        score_arr,
        angular_span_arr,
        contact_norm_arr,
        score_norm_arr,
        angular_span_norm_arr,
    )

    plot_distributions(
        contact_arr,
        score_arr,
        angular_span_arr,
        contact_norm_arr,
        score_norm_arr,
        angular_span_norm_arr,
        dataset_dir,
    )