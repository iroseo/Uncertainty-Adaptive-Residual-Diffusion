"""Release availability checks; no model imports, replacement algorithms or weights."""
import argparse
import json
from pathlib import Path

WITHHELD_FUNCTIONS = (
    "model.sr3_modules.uard.UARDiffusion.q_sample",
    "model.sr3_modules.uard.UARDiffusion.generate",
)
AVAILABLE = frozenset({"s0", "metrics", "prepare"})


def require_available(component):
    if component not in AVAILABLE:
        raise NotImplementedError(
            "This UARD source preview includes model, training and configuration files, "
            "but S1/S2 training and optical generation require two unreleased functions: "
            "UARDiffusion.q_sample and UARDiffusion.generate. "
            "See RELEASE_SCOPE.md. Requested component: " + str(component)
        )


def check_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    stage = cfg.get("stage")
    if stage not in (0, 1, 2):
        raise ValueError("Expected stage=0, 1 or 2 in configuration")
    require_available("s%d" % stage)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("s0", "s1", "s2", "inference", "metrics", "prepare"))
    args = parser.parse_args()
    if args.check:
        try:
            require_available(args.check)
        except NotImplementedError as exc:
            parser.exit(2, str(exc) + "\n")
        print("Release component available: " + args.check + ". Dependencies/data must be configured separately.")
    else:
        print(json.dumps({"release": "source_preview", "available_components": sorted(AVAILABLE),
                          "withheld_functions": WITHHELD_FUNCTIONS,
                          "full_paper_reproduction_available": False}, indent=2))


if __name__ == "__main__":
    main()
