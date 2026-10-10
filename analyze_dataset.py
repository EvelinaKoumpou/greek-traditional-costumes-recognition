# ============================================================================
# analyze_dataset.py
# Τελικός έλεγχος dataset παραδοσιακών φορεσιών
#
# Το script:
# 1. Διαβάζει όλους τους υποφακέλους του dataset.
# 2. Ελέγχει αν οι εικόνες ανοίγουν σωστά.
# 3. Καταγράφει διαστάσεις, format, mode και μέγεθος αρχείου.
# 4. Εντοπίζει ακριβή διπλότυπα με SHA-256.
# 5. Εντοπίζει πιθανώς πολύ όμοιες εικόνες με perceptual dHash.
# 6. Δημιουργεί αναφορές CSV και TXT.
#
# Δεν αλλάζει, δεν μετακινεί και δεν διαγράφει καμία εικόνα.
# ============================================================================

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Μικρή απόσταση σημαίνει μεγαλύτερη οπτική ομοιότητα.
# Για αρχικό έλεγχο, τιμή 6 είναι συντηρητική.
DHASH_DISTANCE_THRESHOLD = 6


def sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Υπολογίζει SHA-256 χωρίς να φορτώνει όλο το αρχείο στη μνήμη."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Υπολογίζει dHash 64-bit.
    Παρόμοιες εικόνες έχουν συνήθως μικρή Hamming distance.
    """
    grayscale = image.convert("L")
    resized = grayscale.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

    pixels = list(resized.getdata())
    bits = []

    for row in range(hash_size):
        row_start = row * (hash_size + 1)

        for col in range(hash_size):
            left_pixel = pixels[row_start + col]
            right_pixel = pixels[row_start + col + 1]
            bits.append(left_pixel > right_pixel)

    hash_value = 0

    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Υπολογίζει πόσα bits διαφέρουν μεταξύ δύο perceptual hashes."""
    return (hash_a ^ hash_b).bit_count()


def find_image_files(dataset_dir: Path) -> list[Path]:
    """Επιστρέφει όλες τις υποστηριζόμενες εικόνες αναδρομικά."""
    return sorted(
        file_path
        for file_path in dataset_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def analyze_image(file_path: Path, dataset_dir: Path) -> dict:
    """Ελέγχει μία εικόνα και επιστρέφει τα χαρακτηριστικά της."""
    relative_path = file_path.relative_to(dataset_dir)
    class_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "UNKNOWN"

    record = {
        "class_name": class_name,
        "filename": file_path.name,
        "relative_path": relative_path.as_posix(),
        "extension": file_path.suffix.lower(),
        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
        "is_valid": False,
        "error": "",
        "format": "",
        "mode": "",
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "sha256": "",
        "dhash_hex": "",
    }

    try:
        # Πρώτος έλεγχος ακεραιότητας.
        with Image.open(file_path) as image:
            image.verify()

        # Ξανανοίγουμε, επειδή το verify() αφήνει το αρχείο μη αναγνώσιμο.
        with Image.open(file_path) as image:
            image.load()

            width, height = image.size
            file_format = image.format or ""

            record.update(
                {
                    "is_valid": True,
                    "format": file_format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4) if height else None,
                    "sha256": sha256_file(file_path),
                    "dhash_hex": f"{difference_hash(image):016x}",
                }
            )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        record["error"] = str(error)

    return record


def build_exact_duplicate_report(valid_df: pd.DataFrame) -> pd.DataFrame:
    """Δημιουργεί αναφορά μόνο για hashes που εμφανίζονται πάνω από μία φορά."""
    duplicate_groups = []

    grouped = valid_df.groupby("sha256", sort=False)

    for group_number, (sha256_value, group) in enumerate(grouped, start=1):
        if len(group) < 2:
            continue

        for _, row in group.iterrows():
            duplicate_groups.append(
                {
                    "duplicate_group": group_number,
                    "sha256": sha256_value,
                    "class_name": row["class_name"],
                    "relative_path": row["relative_path"],
                }
            )

    return pd.DataFrame(
        duplicate_groups,
        columns=["duplicate_group", "sha256", "class_name", "relative_path"],
    )


def build_similar_pairs_report(
    valid_df: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """
    Συγκρίνει εικόνες μέσα στην ίδια κατηγορία.
    Δεν χαρακτηρίζει αυτόματα μια εικόνα ως διπλότυπη· δημιουργεί λίστα για οπτικό έλεγχο.
    """
    rows = []

    for class_name, class_df in valid_df.groupby("class_name"):
        records = class_df.to_dict("records")

        for image_a, image_b in itertools.combinations(records, 2):
            hash_a = int(image_a["dhash_hex"], 16)
            hash_b = int(image_b["dhash_hex"], 16)
            distance = hamming_distance(hash_a, hash_b)

            if distance <= threshold:
                rows.append(
                    {
                        "class_name": class_name,
                        "image_1": image_a["relative_path"],
                        "image_2": image_b["relative_path"],
                        "dhash_distance": distance,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["class_name", "image_1", "image_2", "dhash_distance"],
    ).sort_values(
        by=["class_name", "dhash_distance", "image_1", "image_2"],
        ignore_index=True,
    )


def write_summary(
    report_df: pd.DataFrame,
    exact_duplicates_df: pd.DataFrame,
    similar_pairs_df: pd.DataFrame,
    output_path: Path,
    dataset_dir: Path,
) -> None:
    """Αποθηκεύει αναγνώσιμη συνοπτική αναφορά."""
    valid_df = report_df[report_df["is_valid"]].copy()
    invalid_df = report_df[~report_df["is_valid"]].copy()

    class_counts = valid_df["class_name"].value_counts().sort_index()
    extensions = report_df["extension"].value_counts().sort_index()
    dimensions = (
        valid_df.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    exact_duplicate_group_count = (
        exact_duplicates_df["duplicate_group"].nunique()
        if not exact_duplicates_df.empty
        else 0
    )

    lines = [
        "=" * 72,
        "ΑΝΑΦΟΡΑ ΕΛΕΓΧΟΥ DATASET",
        "=" * 72,
        f"Φάκελος dataset: {dataset_dir.resolve()}",
        f"Σύνολο αρχείων εικόνας: {len(report_df)}",
        f"Έγκυρες εικόνες: {len(valid_df)}",
        f"Μη έγκυρες / κατεστραμμένες: {len(invalid_df)}",
        f"Αριθμός κατηγοριών: {valid_df['class_name'].nunique()}",
        "",
        "ΕΙΚΟΝΕΣ ΑΝΑ ΚΑΤΗΓΟΡΙΑ",
        "-" * 72,
    ]

    for class_name, count in class_counts.items():
        lines.append(f"{class_name}: {count}")

    lines.extend(
        [
            "",
            "ΜΟΡΦΕΣ ΑΡΧΕΙΩΝ",
            "-" * 72,
        ]
    )

    for extension, count in extensions.items():
        lines.append(f"{extension}: {count}")

    lines.extend(
        [
            "",
            "10 ΣΥΧΝΟΤΕΡΕΣ ΔΙΑΣΤΑΣΕΙΣ",
            "-" * 72,
        ]
    )

    for (width, height), count in dimensions.items():
        lines.append(f"{int(width)} x {int(height)}: {count}")

    lines.extend(
        [
            "",
            "ΔΙΠΛΟΤΥΠΑ ΚΑΙ ΟΜΟΙΟΤΗΤΑ",
            "-" * 72,
            f"Ομάδες ακριβών διπλοτύπων: {exact_duplicate_group_count}",
            f"Εικόνες που συμμετέχουν σε ακριβή διπλότυπα: {len(exact_duplicates_df)}",
            (
                "Ζεύγη πιθανώς πολύ όμοιων εικόνων "
                f"(dHash distance <= {DHASH_DISTANCE_THRESHOLD}): {len(similar_pairs_df)}"
            ),
            "",
            "ΣΗΜΑΝΤΙΚΗ ΣΗΜΕΙΩΣΗ",
            "-" * 72,
            (
                "Το possible_similar_pairs.csv είναι λίστα υποψηφίων για οπτικό έλεγχο. "
                "Δεν διαγράφουμε εικόνες αυτόματα, επειδή δύο διαφορετικές λήψεις της ίδιας "
                "φορεσιάς μπορεί να είναι χρήσιμες, αρκεί να παραμείνουν στην ίδια ομάδα "
                "κατά το train/validation/test split."
            ),
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Έλεγχος dataset εικόνων χωρίς καμία τροποποίηση των αρχείων."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Διαδρομή προς τον φάκελο dataset. Προεπιλογή: dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dataset_analysis"),
        help="Φάκελος αποθήκευσης αναφορών.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=DHASH_DISTANCE_THRESHOLD,
        help="Μέγιστη dHash Hamming distance για υποψήφια όμοια ζεύγη.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    dataset_dir: Path = args.dataset
    output_dir: Path = args.output
    similarity_threshold: int = args.similarity_threshold

    if not dataset_dir.exists():
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκε ο φάκελος: {dataset_dir.resolve()}")
        return 1

    if not dataset_dir.is_dir():
        print(f"ΣΦΑΛΜΑ: Η διαδρομή δεν είναι φάκελος: {dataset_dir.resolve()}")
        return 1

    image_files = find_image_files(dataset_dir)

    if not image_files:
        print(f"ΣΦΑΛΜΑ: Δεν βρέθηκαν εικόνες στον φάκελο: {dataset_dir.resolve()}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ΕΛΕΓΧΟΣ DATASET")
    print("=" * 72)
    print(f"Dataset: {dataset_dir.resolve()}")
    print(f"Εικόνες που βρέθηκαν: {len(image_files)}")
    print()

    records = []

    for index, file_path in enumerate(image_files, start=1):
        print(f"[{index:03d}/{len(image_files):03d}] {file_path}")
        records.append(analyze_image(file_path, dataset_dir))

    report_df = pd.DataFrame(records)
    valid_df = report_df[report_df["is_valid"]].copy()

    exact_duplicates_df = build_exact_duplicate_report(valid_df)
    similar_pairs_df = build_similar_pairs_report(
        valid_df=valid_df,
        threshold=similarity_threshold,
    )

    dataset_report_path = output_dir / "dataset_report.csv"
    class_summary_path = output_dir / "class_summary.csv"
    exact_duplicates_path = output_dir / "exact_duplicates.csv"
    similar_pairs_path = output_dir / "possible_similar_pairs.csv"
    summary_path = output_dir / "dataset_summary.txt"

    report_df.to_csv(dataset_report_path, index=False, encoding="utf-8-sig")

    class_summary_df = (
        valid_df.groupby("class_name")
        .agg(
            image_count=("relative_path", "count"),
            min_width=("width", "min"),
            max_width=("width", "max"),
            average_width=("width", "mean"),
            min_height=("height", "min"),
            max_height=("height", "max"),
            average_height=("height", "mean"),
            average_file_size_kb=("file_size_kb", "mean"),
        )
        .reset_index()
        .sort_values("class_name")
    )

    numeric_columns = [
        "average_width",
        "average_height",
        "average_file_size_kb",
    ]
    class_summary_df[numeric_columns] = class_summary_df[numeric_columns].round(2)
    class_summary_df.to_csv(class_summary_path, index=False, encoding="utf-8-sig")

    exact_duplicates_df.to_csv(
        exact_duplicates_path,
        index=False,
        encoding="utf-8-sig",
    )

    similar_pairs_df.to_csv(
        similar_pairs_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_summary(
        report_df=report_df,
        exact_duplicates_df=exact_duplicates_df,
        similar_pairs_df=similar_pairs_df,
        output_path=summary_path,
        dataset_dir=dataset_dir,
    )

    print()
    print("=" * 72)
    print("Ο ΕΛΕΓΧΟΣ ΟΛΟΚΛΗΡΩΘΗΚΕ")
    print("=" * 72)
    print(f"Αναλυτική καταγραφή: {dataset_report_path.resolve()}")
    print(f"Σύνοψη ανά κατηγορία: {class_summary_path.resolve()}")
    print(f"Ακριβή διπλότυπα: {exact_duplicates_path.resolve()}")
    print(f"Πιθανώς όμοια ζεύγη: {similar_pairs_path.resolve()}")
    print(f"Συνοπτική αναφορά: {summary_path.resolve()}")
    print()
    print("Δεν τροποποιήθηκε καμία εικόνα του αρχικού dataset.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
