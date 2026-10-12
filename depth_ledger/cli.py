#!/usr/bin/env python3
"""Depth Ledger CLI — DepthPaper bundles and Gaussian splat lift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from depth_ledger.bundle import bundle_from_image
from depth_ledger.lift import LiftOptions, lift_bundle
from depth_ledger.models import DEFAULT_MODEL, MAX_SIZE, is_promptda, normalize_model_key
from depth_ledger.paths import BUNDLE_CAMERA_NAME, BUNDLE_DEPTH_NAME, BUNDLE_RGB_NAME


def _cmd_bundle(args: argparse.Namespace) -> int:
    image = Path(args.image)
    if not image.is_file():
        print(f"error: image not found: {image}", file=sys.stderr)
        return 1
    out = Path(args.output)
    prompt = Path(args.prompt_depth) if args.prompt_depth else None
    model = normalize_model_key(args.model)
    if is_promptda(model) and prompt is None:
        print("error: PromptDA requires --prompt-depth", file=sys.stderr)
        return 1

    paths = bundle_from_image(
        image,
        out,
        model_key=model,
        max_size=int(args.max_size),
        prompt_depth_path=prompt,
    )
    print(f"wrote {paths['rgb']}")
    print(f"wrote {paths['depth_mm']}")
    print(f"wrote {paths['camera']}")

    if args.lift:
        return _lift_into(out, args)
    return 0


def _lift_into(bundle_dir: Path, args: argparse.Namespace) -> int:
    ply = Path(args.ply) if args.ply else bundle_dir / "splat.ply"
    spz_out = Path(args.spz) if args.spz else bundle_dir / "splat.spz"
    opts = LiftOptions(
        stride=int(args.stride),
        depth_min_m=float(args.depth_min),
        depth_max_m=float(args.depth_max),
    )
    try:
        ply_path, spz_path = lift_bundle(bundle_dir, ply, spz_out, opts)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {ply_path}")
    print(f"wrote {spz_path}")
    return 0


def _cmd_lift(args: argparse.Namespace) -> int:
    bundle_dir = Path(args.bundle)
    for name in (BUNDLE_RGB_NAME, BUNDLE_DEPTH_NAME, BUNDLE_CAMERA_NAME):
        if not (bundle_dir / name).is_file():
            print(f"error: missing {bundle_dir / name}", file=sys.stderr)
            return 1
    return _lift_into(bundle_dir, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-ledger",
        description="Metric depth (Depth Pro / PromptDA) → DepthPaper bundle → 3DGS splat.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bundle = sub.add_parser("bundle", help="Infer depth and write a DepthPaper folder")
    bundle.add_argument("image", type=Path, help="Input RGB image")
    bundle.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory (DepthPaper bundle)",
    )
    bundle.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="depth-pro | promptda-small | promptda-large",
    )
    bundle.add_argument("--prompt-depth", type=Path, default=None, help="PromptDA uint16 mm PNG")
    bundle.add_argument("--max-size", type=int, default=MAX_SIZE, help="PromptDA max side")
    bundle.add_argument("--lift", action="store_true", help="Also write splat.ply and splat.spz")
    bundle.add_argument("--ply", type=Path, default=None, help="PLY path when --lift")
    bundle.add_argument("--spz", type=Path, default=None, help="SPZ path when --lift")
    bundle.add_argument("--stride", type=int, default=2, help="Lift subsample stride")
    bundle.add_argument("--depth-min", type=float, default=0.05, help="Lift near clip (m)")
    bundle.add_argument("--depth-max", type=float, default=80.0, help="Lift far clip (m)")
    bundle.set_defaults(func=_cmd_bundle)

    lift = sub.add_parser("lift", help="Lift an existing DepthPaper bundle to .ply + .spz")
    lift.add_argument("bundle", type=Path, help="DepthPaper bundle directory")
    lift.add_argument("--ply", type=Path, default=None, help="Output PLY (default: bundle/splat.ply)")
    lift.add_argument("--spz", type=Path, default=None, help="Output SPZ (default: bundle/splat.spz)")
    lift.add_argument("--stride", type=int, default=2)
    lift.add_argument("--depth-min", type=float, default=0.05)
    lift.add_argument("--depth-max", type=float, default=80.0)
    lift.set_defaults(func=_cmd_lift)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
