import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATOR = SCRIPT_DIR / "video_crop_calibrator_fixed.py"
DEFAULT_CONFIG = SCRIPT_DIR / "crop_calibration_space.json"
DEFAULT_SRC = Path("/home/cra-space-center/Desktop/real-6-day-26-30-copy/non-flip")
DEFAULT_DST = Path("/home/cra-space-center/Desktop/real-6-day-26-30-copy/flip")


def load_calibrator_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("video_crop_calibrator_fixed", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load calibrator module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_videos(src_dir: Path, recursive: bool):
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(p for p in src_dir.glob(pattern) if p.is_file())


def build_default_config(module, video_path: Path):
    cap = module.cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")

    h, w = frame.shape[:2]
    return {
        "left_x": 0,
        "left_y": 0,
        "left_w": w // 2,
        "left_h": h,
        "right_x": w // 2,
        "right_y": 0,
        "right_w": min(460, w - (w // 2)),
        "right_h": h,
        "right_rotate_180": True,
        "gap_px": 0,
        "preview_scale": 50,
        "seam_match_enable": False,
        "seam_width_px": 30,
        "right_luma_gain": 1.0,
    }


def load_batch_config(module, config_path: Path | None, sample_video: Path):
    default_config = build_default_config(module, sample_video)

    if config_path is None:
        print("No config JSON provided. Falling back to auto-generated default calibration.")
        return default_config

    cap = module.cv2.VideoCapture(str(sample_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for config sizing: {sample_video}")

    frame_w = int(cap.get(module.cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(module.cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return module.load_config_if_exists(str(config_path), default_config, frame_w, frame_h)


def output_path_for(src_file: Path, src_root: Path, dst_root: Path) -> Path:
    rel = src_file.relative_to(src_root)
    return dst_root / rel


def process_videos(module, videos, src_root: Path, dst_root: Path, config, args):
    success = 0
    failures = []

    for index, video_path in enumerate(videos, start=1):
        out_path = output_path_for(video_path, src_root, dst_root)

        if args.skip_existing and out_path.exists():
            print(f"[{index}/{len(videos)}] Skip existing: {out_path}")
            continue

        print(f"[{index}/{len(videos)}] Processing: {video_path}")
        print(f"           Output: {out_path}")

        try:
            module.save_processed_video_ffmpeg(
                input_video=str(video_path),
                output_video=str(out_path),
                p=config,
                max_seconds=args.max_seconds,
                crf=args.crf,
                preset=args.preset,
            )
            success += 1
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found in PATH. Please install ffmpeg before running batch export.") from exc
        except Exception as exc:
            failures.append((video_path, str(exc)))
            print(f"           FAILED: {exc}")

    return success, failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-process videos from non-flip to flip using video_crop_calibrator_fixed.py"
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source directory containing input videos")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Destination directory for processed videos")
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=DEFAULT_CALIBRATOR,
        help="Path to video_crop_calibrator_fixed.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Calibration JSON created from the interactive calibrator. Defaults to crop_calibration_space.json.",
    )
    parser.add_argument("--max-seconds", type=float, default=None, help="Limit each output video duration in seconds")
    parser.add_argument("--crf", type=int, default=12, help="FFmpeg CRF quality. Lower = better/larger")
    parser.add_argument("--preset", default="slow", help="FFmpeg preset, e.g. medium/slow/slower")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist in destination")
    parser.add_argument("--no-recursive", action="store_true", help="Only process mp4 files in the top level of src")
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N discovered videos")
    parser.add_argument("--print-config", action="store_true", help="Print the calibration config before processing")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.src.exists():
        raise RuntimeError(f"Source directory does not exist: {args.src}")
    if not args.calibrator.exists():
        raise RuntimeError(f"Calibrator script does not exist: {args.calibrator}")

    module = load_calibrator_module(args.calibrator)

    videos = discover_videos(args.src, recursive=not args.no_recursive)
    if args.max_files is not None:
        videos = videos[: args.max_files]

    if not videos:
        print(f"No .mp4 files found in: {args.src}")
        return

    config = load_batch_config(module, args.config, videos[0])
    if args.print_config:
        print(json.dumps(config, indent=2))

    args.dst.mkdir(parents=True, exist_ok=True)

    success, failures = process_videos(module, videos, args.src, args.dst, config, args)

    print("")
    print("Batch complete")
    print(f"  Source videos : {len(videos)}")
    print(f"  Success       : {success}")
    print(f"  Failed        : {len(failures)}")

    if failures:
        print("")
        print("Failed files:")
        for video_path, reason in failures:
            print(f"  - {video_path}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
