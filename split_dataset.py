# ============================================================================
# split_dataset.py
# Group-aware διαχωρισμός dataset σε train / validation / test
#
# Στόχοι:
# 1. Δεν τροποποιεί ποτέ τον αρχικό φάκελο dataset.
# 2. Αντιγράφει τις εικόνες σε νέο φάκελο dataset_split.
# 3. Κρατά παρόμοιες εικόνες της ίδιας ομάδας στο ίδιο subset, ώστε να
#    αποφεύγεται data leakage.
# 4. Χρησιμοποιεί σταθερό random seed για αναπαραγωγιμότητα.
# 5. Δημιουργεί αναλυτικές αναφορές CSV και TXT.
#
# Προεπιλεγμένες αναλογίες:
# train      70%
# validation 15%
# test       15%
# ============================================================================

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ----------------------------------------------------------------------------
# Union-Find: ενώνει ζεύγη πιθανώς όμοιων εικόνων στην ίδια ομάδα.
# ----------------------------------------------------------------------------
class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item_a: str, item_b: str) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


@dataclass
class ImageGroup:
    class_name: str
    group_id: str
    relative_paths: list[str]

    @property
    def size(self) -> int:
        return len(self.relative_paths)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Διαχωρισμός dataset σε train/validation/test, "
            "με ομαδοποίηση πιθανώς όμοιων εικόνων."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Αρχικός φάκελος dataset. Προεπιλογή: dataset",
    )

    parser.add_argument(
        "--similar-pairs",
        type=Path,
        default=Path("reports/dataset_analysis/possible_similar_pairs.csv"),
        help="CSV με πιθανά όμοια ζεύγη εικόνων.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_split"),
        help="Νέος φάκελος εξόδου. Προεπιλογή: dataset_split",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/dataset_split"),
        help="Φάκελος αναφορών.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Σταθερό seed για αναπαραγωγιμότητα.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Διαγράφει και ξαναδημιουργεί το dataset_split, αν υπάρχει ήδη.",
    )

    return parser.parse_args()


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Οι αναλογίες πρέπει να έχουν άθροισμα 1.0, αλλά είναι {total:.6f}."
        )

    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Όλες οι αναλογίες πρέπει να είναι μεγαλύτερες από 0.")


def discover_images(dataset_dir: Path) -> dict[str, list[str]]:
    """
    Επιστρέφει:
        {
            "thraki": ["thraki/image1.jpg", ...],
            ...
        }
    """
    class_to_images: dict[str, list[str]] = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        relative_paths = sorted(
            file_path.relative_to(dataset_dir).as_posix()
            for file_path in class_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if relative_paths:
            class_to_images[class_dir.name] = relative_paths

    return class_to_images


def normalize_relative_path(path_value: str, dataset_dir_name: str) -> str:
    """
    Μετατρέπει:
      dataset/thraki/a.jpg -> thraki/a.jpg
      thraki/a.jpg         -> thraki/a.jpg
    """
    normalized = Path(str(path_value).replace("\\", "/")).as_posix().lstrip("./")

    prefix = f"{dataset_dir_name}/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]

    return normalized


def load_similar_pairs(
    csv_path: Path,
    dataset_dir_name: str,
) -> list[tuple[str, str]]:
    if not csv_path.exists():
        print(
            f"ΠΡΟΕΙΔΟΠΟΙΗΣΗ: Δεν βρέθηκε το {csv_path}. "
            "Ο διαχωρισμός θα γίνει χωρίς ομαδοποίηση όμοιων εικόνων."
        )
        return []

    dataframe = pd.read_csv(csv_path)

    required_columns = {"image_1", "image_2"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Λείπουν στήλες από το {csv_path}: {sorted(missing_columns)}"
        )

    pairs: list[tuple[str, str]] = []

    for _, row in dataframe.iterrows():
        image_1 = normalize_relative_path(row["image_1"], dataset_dir_name)
        image_2 = normalize_relative_path(row["image_2"], dataset_dir_name)
        pairs.append((image_1, image_2))

    return pairs


def build_groups_for_class(
    class_name: str,
    image_paths: list[str],
    similar_pairs: list[tuple[str, str]],
) -> list[ImageGroup]:
    image_set = set(image_paths)
    union_find = UnionFind(image_paths)

    for image_1, image_2 in similar_pairs:
        if image_1 in image_set and image_2 in image_set:
            union_find.union(image_1, image_2)

    grouped_paths: dict[str, list[str]] = defaultdict(list)

    for image_path in image_paths:
        grouped_paths[union_find.find(image_path)].append(image_path)

    groups = []

    for group_number, paths in enumerate(
        sorted(grouped_paths.values(), key=lambda values: (len(values), values[0])),
        start=1,
    ):
        groups.append(
            ImageGroup(
                class_name=class_name,
                group_id=f"{class_name}_group_{group_number:03d}",
                relative_paths=sorted(paths),
            )
        )

    return groups


def allocate_groups(
    groups: list[ImageGroup],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[ImageGroup]]:
    """
    Greedy κατανομή ομάδων με στόχο τις επιθυμητές αναλογίες.
    Οι μεγαλύτερες ομάδες τοποθετούνται πρώτες.
    """
    total_images = sum(group.size for group in groups)

    target_counts = {
        "train": total_images * train_ratio,
        "validation": total_images * validation_ratio,
        "test": total_images * test_ratio,
    }

    allocations: dict[str, list[ImageGroup]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    shuffled_groups = groups[:]
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda group: group.size, reverse=True)

    # Προσπαθούμε πρώτα να δώσουμε τουλάχιστον μία ομάδα σε κάθε subset.
    subset_order = ["train", "validation", "test"]

    for subset_name, group in zip(subset_order, shuffled_groups[:3]):
        allocations[subset_name].append(group)
        current_counts[subset_name] += group.size

    remaining_groups = shuffled_groups[3:]

    for group in remaining_groups:
        best_subset = min(
            subset_order,
            key=lambda subset_name: (
                current_counts[subset_name] / target_counts[subset_name]
                if target_counts[subset_name] > 0
                else float("inf"),
                current_counts[subset_name],
            ),
        )

        allocations[best_subset].append(group)
        current_counts[best_subset] += group.size

    return allocations


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Ο φάκελος {output_dir.resolve()} υπάρχει ήδη.\n"
                "Για να ξαναδημιουργηθεί, εκτέλεσε το script με --overwrite."
            )

        shutil.rmtree(output_dir)

    for subset_name in ("train", "validation", "test"):
        (output_dir / subset_name).mkdir(parents=True, exist_ok=True)


def copy_allocations(
    dataset_dir: Path,
    output_dir: Path,
    allocations_by_class: dict[str, dict[str, list[ImageGroup]]],
) -> list[dict]:
    manifest_rows: list[dict] = []

    for class_name, subset_allocations in sorted(allocations_by_class.items()):
        for subset_name, groups in subset_allocations.items():
            destination_class_dir = output_dir / subset_name / class_name
            destination_class_dir.mkdir(parents=True, exist_ok=True)

            for group in groups:
                for relative_path in group.relative_paths:
                    source_path = dataset_dir / relative_path
                    destination_path = destination_class_dir / Path(relative_path).name

                    if not source_path.exists():
                        raise FileNotFoundError(
                            f"Δεν βρέθηκε η εικόνα: {source_path.resolve()}"
                        )

                    if destination_path.exists():
                        raise FileExistsError(
                            "Υπάρχουν δύο εικόνες με το ίδιο όνομα μέσα στην ίδια "
                            f"κατηγορία: {destination_path}"
                        )

                    shutil.copy2(source_path, destination_path)

                    manifest_rows.append(
                        {
                            "subset": subset_name,
                            "class_name": class_name,
                            "group_id": group.group_id,
                            "group_size": group.size,
                            "source_relative_path": relative_path,
                            "destination_relative_path": (
                                destination_path.relative_to(output_dir).as_posix()
                            ),
                        }
                    )

    return manifest_rows


def create_reports(
    manifest_df: pd.DataFrame,
    report_dir: Path,
    ratios: dict[str, float],
    seed: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "split_manifest.csv"
    class_distribution_path = report_dir / "split_class_distribution.csv"
    summary_path = report_dir / "split_summary.txt"

    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    class_distribution_df = (
        manifest_df.groupby(["class_name", "subset"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for subset_name in ("train", "validation", "test"):
        if subset_name not in class_distribution_df.columns:
            class_distribution_df[subset_name] = 0

    class_distribution_df["total"] = (
        class_distribution_df["train"]
        + class_distribution_df["validation"]
        + class_distribution_df["test"]
    )

    class_distribution_df = class_distribution_df[
        ["class_name", "train", "validation", "test", "total"]
    ]

    class_distribution_df.to_csv(
        class_distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    subset_counts = manifest_df["subset"].value_counts()
    total_images = len(manifest_df)
    grouped_image_count = (
        manifest_df.loc[manifest_df["group_size"] > 1, "source_relative_path"].count()
    )
    multi_image_group_count = manifest_df.loc[
        manifest_df["group_size"] > 1, "group_id"
    ].nunique()

    lines = [
        "=" * 72,
        "ΑΝΑΦΟΡΑ TRAIN / VALIDATION / TEST SPLIT",
        "=" * 72,
        f"Random seed: {seed}",
        f"Συνολικές εικόνες: {total_images}",
        "",
        "ΕΠΙΘΥΜΗΤΕΣ ΑΝΑΛΟΓΙΕΣ",
        "-" * 72,
        f"train: {ratios['train']:.2%}",
        f"validation: {ratios['validation']:.2%}",
        f"test: {ratios['test']:.2%}",
        "",
        "ΠΡΑΓΜΑΤΙΚΗ ΚΑΤΑΝΟΜΗ",
        "-" * 72,
    ]

    for subset_name in ("train", "validation", "test"):
        count = int(subset_counts.get(subset_name, 0))
        percentage = count / total_images if total_images else 0.0
        lines.append(f"{subset_name}: {count} ({percentage:.2%})")

    lines.extend(
        [
            "",
            "ΟΜΑΔΟΠΟΙΗΣΗ ΟΜΟΙΩΝ ΕΙΚΟΝΩΝ",
            "-" * 72,
            f"Ομάδες με περισσότερες από μία εικόνες: {multi_image_group_count}",
            f"Εικόνες που συμμετέχουν σε τέτοιες ομάδες: {grouped_image_count}",
            "",
            "ΕΛΕΓΧΟΣ DATA LEAKAGE",
            "-" * 72,
            (
                "Κάθε group_id εμφανίζεται αποκλειστικά σε ένα subset. "
                "Έτσι οι πιθανώς όμοιες εικόνες μιας ομάδας δεν μοιράζονται "
                "ανάμεσα σε train, validation και test."
            ),
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Αναφορές:")
    print(f"- {manifest_path.resolve()}")
    print(f"- {class_distribution_path.resolve()}")
    print(f"- {summary_path.resolve()}")


def verify_no_group_leakage(manifest_df: pd.DataFrame) -> None:
    leakage_check = manifest_df.groupby("group_id")["subset"].nunique()
    leaking_groups = leakage_check[leakage_check > 1]

    if not leaking_groups.empty:
        raise RuntimeError(
            "Εντοπίστηκε data leakage: κάποιες ομάδες εμφανίζονται σε πάνω "
            "από ένα subset."
        )


def main() -> int:
    args = parse_arguments()

    try:
        validate_ratios(
            args.train_ratio,
            args.validation_ratio,
            args.test_ratio,
        )

        dataset_dir: Path = args.dataset
        output_dir: Path = args.output
        report_dir: Path = args.report_dir

        if not dataset_dir.exists() or not dataset_dir.is_dir():
            print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε ο φάκελος {dataset_dir.resolve()}")
            return 1

        class_to_images = discover_images(dataset_dir)

        if not class_to_images:
            print("ΣΦΑΛΜΑ: Δεν βρέθηκαν κατηγορίες και εικόνες στο dataset.")
            return 1

        total_images = sum(len(paths) for paths in class_to_images.values())

        similar_pairs = load_similar_pairs(
            csv_path=args.similar_pairs,
            dataset_dir_name=dataset_dir.name,
        )

        prepare_output_directory(
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

        print("=" * 72)
        print("ΔΗΜΙΟΥΡΓΙΑ TRAIN / VALIDATION / TEST SPLIT")
        print("=" * 72)
        print(f"Dataset: {dataset_dir.resolve()}")
        print(f"Κατηγορίες: {len(class_to_images)}")
        print(f"Εικόνες: {total_images}")
        print(f"Similar pairs που διαβάστηκαν: {len(similar_pairs)}")
        print(f"Random seed: {args.seed}")
        print()

        rng = random.Random(args.seed)
        allocations_by_class: dict[str, dict[str, list[ImageGroup]]] = {}

        for class_name, image_paths in sorted(class_to_images.items()):
            groups = build_groups_for_class(
                class_name=class_name,
                image_paths=image_paths,
                similar_pairs=similar_pairs,
            )

            allocations = allocate_groups(
                groups=groups,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
                rng=rng,
            )

            allocations_by_class[class_name] = allocations

            counts = {
                subset_name: sum(group.size for group in subset_groups)
                for subset_name, subset_groups in allocations.items()
            }

            print(
                f"{class_name:<20} "
                f"train={counts['train']:<3} "
                f"validation={counts['validation']:<3} "
                f"test={counts['test']:<3} "
                f"total={len(image_paths)}"
            )

        manifest_rows = copy_allocations(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            allocations_by_class=allocations_by_class,
        )

        manifest_df = pd.DataFrame(manifest_rows)
        verify_no_group_leakage(manifest_df)

        copied_count = len(manifest_df)

        if copied_count != total_images:
            raise RuntimeError(
                f"Αναμένονταν {total_images} εικόνες, αλλά αντιγράφηκαν {copied_count}."
            )

        create_reports(
            manifest_df=manifest_df,
            report_dir=report_dir,
            ratios={
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
            seed=args.seed,
        )

        print()
        print("=" * 72)
        print("Ο ΔΙΑΧΩΡΙΣΜΟΣ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
        print("=" * 72)
        print(f"Νέο dataset: {output_dir.resolve()}")
        print(f"Αντιγράφηκαν συνολικά {copied_count} εικόνες.")
        print("Ο αρχικός φάκελος dataset δεν τροποποιήθηκε.")

        return 0

    except Exception as error:
        print()
        print("=" * 72)
        print("Ο ΔΙΑΧΩΡΙΣΜΟΣ ΔΕΝ ΟΛΟΚΛΗΡΩΘΗΚΕ")
        print("=" * 72)
        print(f"Σφάλμα: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
