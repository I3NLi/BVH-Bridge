from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .canonical import common_semantics, compare_canonical
from .formats import FORMAT_NAMES
from .pipeline import convert_bvh, format_summary, load_canonical


def cmd_inspect(args: argparse.Namespace) -> None:
    print(json.dumps(format_summary(args.input), ensure_ascii=False, indent=2, sort_keys=True))


def cmd_convert(args: argparse.Namespace) -> None:
    out = convert_bvh(
        args.input,
        args.output,
        args.target,
        source_format=args.source,
        template_path=args.template,
        max_frames=args.max_frames,
    )
    print(f"[convert] {args.input} -> {out} target={args.target}")


def _sample_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "lafan1": args.lafan,
        "nokov": args.nokov,
        "wbody": args.wbody,
    }


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: frames={metrics['frames']:.0f} joints={metrics['joints']:.0f} "
        f"pos_mean={metrics['pos_mean']:.4f} pos_p95={metrics['pos_p95']:.4f} "
        f"pos_max={metrics['pos_max']:.4f} root_mean={metrics['root_mean']:.4f} "
        f"rot_mean={metrics['rot_mean_deg']:.2f}deg rot_p95={metrics['rot_p95_deg']:.2f}deg"
    )


def cmd_validate(args: argparse.Namespace) -> None:
    samples = _sample_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    canonical = {
        name: load_canonical(path, source_format=name, max_frames=args.max_frames)
        for name, path in samples.items()
    }

    rows: list[dict[str, object]] = []
    for source, target in itertools.permutations(FORMAT_NAMES, 2):
        output = args.out_dir / f"{source}_to_{target}.bvh"
        convert_bvh(samples[source], output, target, source_format=source, max_frames=args.max_frames)
        converted = load_canonical(output, source_format=target, max_frames=args.max_frames)
        metrics = compare_canonical(canonical[source], converted)
        label = f"{source}->{target}"
        _print_metrics(label, metrics)
        rows.append({"path": str(output), "case": label, **metrics})

    for source, mid in itertools.permutations(FORMAT_NAMES, 2):
        first = args.out_dir / f"closed_{source}_to_{mid}.bvh"
        final = args.out_dir / f"closed_{source}_to_{mid}_to_{source}.bvh"
        convert_bvh(samples[source], first, mid, source_format=source, max_frames=args.max_frames)
        convert_bvh(first, final, source, source_format=mid)
        closed = load_canonical(final, source_format=source, max_frames=args.max_frames)
        path_names = common_semantics(canonical[source], canonical[mid])
        lost_names = sorted(set(canonical[source].positions) - set(path_names))
        metrics = compare_canonical(canonical[source], closed, names=path_names)
        label = f"{source}->{mid}->{source}"
        _print_metrics(label, metrics)
        rows.append({"path": str(final), "case": label, "lost_joints": lost_names, **metrics})

    report = args.out_dir / "validation_report.jsonl"
    report.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"[report] {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bvh_bridge", description="Convert LaFAN1/Nokov/Wbody BVH through a canonical format.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect = sub.add_parser("inspect", help="Detect format and print canonical summary")
    inspect.add_argument("input", type=Path)
    inspect.set_defaults(func=cmd_inspect)

    convert = sub.add_parser("convert", help="Convert a BVH to a target format")
    convert.add_argument("--input", "-i", type=Path, required=True)
    convert.add_argument("--output", "-o", type=Path, required=True)
    convert.add_argument("--source", default="auto", choices=("auto", *FORMAT_NAMES))
    convert.add_argument("--target", required=True, choices=FORMAT_NAMES)
    convert.add_argument("--template", type=Path, default=None, help="Optional target BVH template")
    convert.add_argument("--max-frames", type=int, default=None)
    convert.set_defaults(func=cmd_convert)

    validate = sub.add_parser("validate", help="Run pairwise and closed-loop validation")
    validate.add_argument("--lafan", type=Path, required=True)
    validate.add_argument("--nokov", type=Path, required=True)
    validate.add_argument("--wbody", type=Path, required=True)
    validate.add_argument("--out-dir", type=Path, required=True)
    validate.add_argument("--max-frames", type=int, default=300)
    validate.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
