"""Audit a complete prediction/target image pair set without computing metrics.

Python 3.8+ and Pillow are sufficient. Images are matched by their exact filename
stems in two flat directories. No filename rewriting, cropping, resizing, or
image-file modifications are performed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings


IMAGE_EXTENSIONS = frozenset((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


class AuditError(Exception):
    """An input set is incomplete, ambiguous, or unreadable."""


def existing_directory(value, label):
    path = Path(value).expanduser()
    if path.is_symlink():
        raise AuditError("{} must not be a symbolic link: {}".format(label, path))
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditError("Cannot resolve {} {}: {}".format(label, path, exc))
    if not path.is_dir():
        raise AuditError("{} is not a directory: {}".format(label, path))
    return path


def is_inside(path, directory):
    # Path.is_relative_to is unavailable on Python 3.8.
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def checked_output(value, pred_root, gt_dir):
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists() or path.is_symlink():
        raise AuditError("Refusing to overwrite an existing output: {}".format(path))
    try:
        path = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise AuditError("Cannot resolve output {}: {}".format(path, exc))
    if is_inside(path, pred_root) or is_inside(path, gt_dir):
        raise AuditError("Output must be outside both image directories: {}".format(path))
    if not path.parent.is_dir():
        raise AuditError("Output parent directory does not exist: {}".format(path.parent))
    return path


def abbreviated(names):
    names = sorted(names)
    text = ", ".join(repr(name) for name in names[:10])
    if len(names) > 10:
        text += ", ... ({} total)".format(len(names))
    return text


def index_images(directory, label):
    by_stem = {}
    portable_names = {}
    unsupported = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise AuditError("{} contains a symbolic link: {}".format(label, path.name))
        if not path.is_file():
            raise AuditError("{} must be flat and contain only image files: {}".format(
                label, path.name))
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            unsupported.append(path.name)
            continue
        stem = path.stem
        if stem in by_stem:
            raise AuditError("Duplicate {} stem {!r}: {} and {}".format(
                label, stem, by_stem[stem].name, path.name))
        folded = stem.casefold()
        if folded in portable_names:
            raise AuditError("Case-only {} stem collision: {!r} and {!r}. "
                             "Use unambiguous names across operating systems.".format(
                                 label, portable_names[folded], stem))
        by_stem[stem] = path
        portable_names[folded] = stem
    if unsupported:
        raise AuditError("Unsupported files in {} (nothing was silently skipped): {}. "
                         "Supported extensions: {}".format(
                             label, abbreviated(unsupported),
                             ", ".join(sorted(IMAGE_EXTENSIONS))))
    if not by_stem:
        raise AuditError("No supported image files in {}: {}".format(label, directory))
    return by_stem


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path, image_module, include_hash):
    try:
        before = path.stat()
        # Decode every pixel. Opening a header alone does not catch truncated
        # images. The RGB conversion is in memory, matching evaluator loading.
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            with image_module.open(path) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise AuditError("Multi-frame images are unsupported: {}".format(path))
                original_mode = source.mode
                image_format = source.format
                source.load()
                with source.convert("RGB") as rgb:
                    size = list(rgb.size)
                    if size[0] <= 0 or size[1] <= 0:
                        raise AuditError("Invalid image dimensions: {}".format(path))
        record = {"file": path.name, "size": size, "original_mode": original_mode,
                  "format": image_format, "rgb_mode": "RGB", "bytes": before.st_size}
        if include_hash:
            record["sha256"] = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AuditError("Image changed during the audit; rerun after writes finish: {}".format(path))
        return record
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("Cannot fully decode image {}: {}: {}".format(
            path, type(exc).__name__, exc))


def audit(args):
    pred_root = existing_directory(args.pred_root, "prediction directory")
    gt_dir = existing_directory(args.gt_dir, "GT directory")
    if pred_root == gt_dir:
        raise AuditError("Prediction and GT directories must be different.")
    output = checked_output(args.output, pred_root, gt_dir)
    predictions = index_images(pred_root, "prediction directory")
    targets = index_images(gt_dir, "GT directory")
    prediction_stems, target_stems = set(predictions), set(targets)
    missing_gt = prediction_stems - target_stems
    missing_predictions = target_stems - prediction_stems
    if missing_gt or missing_predictions:
        messages = []
        if missing_gt:
            messages.append("Predictions without GT: " + abbreviated(missing_gt))
        if missing_predictions:
            messages.append("GT without predictions: " + abbreviated(missing_predictions))
        raise AuditError("Pair sets are not a bijection. " + "; ".join(messages))
    if args.expected_count is not None and len(predictions) != args.expected_count:
        raise AuditError("Expected {} pairs, found {}.".format(
            args.expected_count, len(predictions)))

    try:
        from PIL import Image, __version__ as pillow_version
    except ImportError:
        raise AuditError("Pillow is required to decode images. Install it with: python -m pip install Pillow")

    records = []
    for stem in sorted(predictions):
        prediction = inspect_image(predictions[stem], Image, args.sha256)
        target = inspect_image(targets[stem], Image, args.sha256)
        if prediction["size"] != target["size"]:
            raise AuditError("Dimension mismatch for {!r}: prediction {} vs GT {}. "
                             "No resizing was applied.".format(
                                 stem, prediction["size"], target["size"]))
        records.append({"stem": stem, "prediction": prediction, "ground_truth": target})

    report = {
        "schema_version": 1,
        "complete": True,
        "audit_only": True,
        "metrics_computed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_directory": str(pred_root),
        "ground_truth_directory": str(gt_dir),
        "pairing": "exact case-sensitive filename stem; complete bijection; flat directories",
        "image_handling": "fully decoded and converted to RGB in memory; no resize, crop, or file modification",
        "count": len(records),
        "expected_count": args.expected_count,
        "pillow_version": pillow_version,
        "sha256_included": args.sha256,
        "pairs": records,
    }
    if output is not None:
        # Exclusive creation also guards against an intervening writer.
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        except OSError as exc:
            raise AuditError("Cannot create new report {}: {}".format(output, exc))
    return report, output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", required=True, help="Flat prediction image directory")
    parser.add_argument("--gt-dir", required=True, help="Flat ground-truth image directory")
    parser.add_argument("--expected-count", type=int, help="Require this exact positive pair count")
    parser.add_argument("--output", help="New JSON report file outside both image directories; parent must exist")
    parser.add_argument("--sha256", action="store_true", help="Record the SHA-256 of every original image file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_count is not None and args.expected_count < 1:
        parser.error("--expected-count must be positive")
    try:
        report, output = audit(args)
    except (AuditError, OSError, RuntimeError) as exc:
        print("Pair audit failed: {}".format(exc), file=sys.stderr)
        return 1
    print("Pair audit passed: {} exact pairs; all images decode to matching RGB dimensions.".format(
        report["count"]))
    print("Prediction: {}".format(report["prediction_directory"]))
    print("GT: {}".format(report["ground_truth_directory"]))
    print("SHA-256: {} | Metrics: not computed".format("included" if args.sha256 else "not requested"))
    if output is not None:
        print("Report: {}".format(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
