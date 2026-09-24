import cv2
import json
import argparse
from pathlib import Path
import subprocess
import numpy as np


def compute_luma(img):
    # img is BGR from OpenCV
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    r = img[:, :, 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def robust_median_luma(strip):
    y = compute_luma(strip)

    # Use percentile clipping to reduce effect from specular highlights/dark noise
    lo, hi = np.percentile(y, [5, 95])
    y_clip = y[(y >= lo) & (y <= hi)]

    if y_clip.size == 0:
        return float(np.median(y))

    return float(np.median(y_clip))


def compute_seam_luma_gain(left, right, seam_width_px=30, min_gain=0.90, max_gain=1.10):
    seam_width_px = max(1, int(seam_width_px))

    sw_left = min(seam_width_px, left.shape[1])
    sw_right = min(seam_width_px, right.shape[1])

    left_strip = left[:, -sw_left:]
    right_strip = right[:, :sw_right]

    left_y = robust_median_luma(left_strip)
    right_y = robust_median_luma(right_strip)

    if right_y < 1.0:
        gain = 1.0
    else:
        gain = left_y / right_y

    gain = max(min_gain, min(max_gain, gain))

    return gain, left_y, right_y

def apply_luma_gain_bgr(img, gain):
    gain = float(gain)

    out = img.astype(np.float32)
    out *= gain
    out = np.clip(out, 0, 255)

    return out.astype(np.uint8)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def normalized_crop_params(c: dict, frame_w: int, frame_h: int) -> dict:
    frame_w = max(1, int(frame_w))
    frame_h = max(1, int(frame_h))

    left_x = clamp(int(c["left_x"]), 0, frame_w - 1)
    left_y = clamp(int(c["left_y"]), 0, frame_h - 1)
    left_w = clamp(int(c["left_w"]), 1, frame_w - left_x)
    left_h = clamp(int(c["left_h"]), 1, frame_h - left_y)

    right_x = clamp(int(c["right_x"]), 0, frame_w - 1)
    right_y = clamp(int(c["right_y"]), 0, frame_h - 1)
    right_w = clamp(int(c["right_w"]), 1, frame_w - right_x)
    right_h = clamp(int(c["right_h"]), 1, frame_h - right_y)

    return {
        "left_x": left_x,
        "left_y": left_y,
        "left_w": left_w,
        "left_h": left_h,
        "right_x": right_x,
        "right_y": right_y,
        "right_w": right_w,
        "right_h": right_h,
        "stack_h": min(left_h, right_h),
    }

def ffmpeg_filter_from_calib(c: dict, frame_w: int, frame_h: int) -> str:
    crop = normalized_crop_params(c, frame_w, frame_h)
    h = crop["stack_h"]

    if c.get("seam_match_enable", False):
        gain = float(c.get("right_luma_gain", 1.0))
        gain = max(0.90, min(1.10, gain))
    else:
        gain = 1.0

    left = (
        f"[0:v]crop={crop['left_w']}:{h}:"
        f"{crop['left_x']}:{crop['left_y']}[left]"
    )

    right = (
        f"[0:v]crop={crop['right_w']}:{h}:"
        f"{crop['right_x']}:{crop['right_y']}[right]"
    )

    if c.get("right_rotate_180", True):
        right_proc = (
            f"[right]hflip,vflip,"
            f"lutrgb=r='clip(val*{gain},0,255)':"
            f"g='clip(val*{gain},0,255)':"
            f"b='clip(val*{gain},0,255)'"
            f"[right_rot]"
        )
    else:
        right_proc = (
            f"[right]lutrgb=r='clip(val*{gain},0,255)':"
            f"g='clip(val*{gain},0,255)':"
            f"b='clip(val*{gain},0,255)'"
            f"[right_rot]"
        )

    gap_px = int(c.get("gap_px", 0))

    if gap_px > 0:
        stack = (
            f"color=c=white:s={gap_px}x{h}[gap];"
            f"[left][gap][right_rot]hstack=inputs=3[stacked]"
        )
    else:
        stack = "[left][right_rot]hstack=inputs=2[stacked]"

    # Pad to even width/height for libx264 + yuv420p.
    # This preserves full crop content and only adds 0 or 1 px if needed.
    even_pad = (
        "[stacked]"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black"
        "[out]"
    )

    return ";".join([left, right, right_proc, stack, even_pad])


def save_processed_video_ffmpeg(input_video, output_video, p, max_seconds=None, crf=0, preset="veryslow"):
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for export size detection: {input_video}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError(f"Cannot detect video dimensions: {input_video}")

    filters = ffmpeg_filter_from_calib(p, frame_w, frame_h)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-filter_complex", filters,
        "-map", "[out]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
    ]

    if max_seconds is not None and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]

    cmd.append(str(output_path))

    print("Running FFmpeg full-quality export:")
    print(" ".join(cmd))

    p_run = subprocess.run(cmd, capture_output=True, text=True)

    if p_run.returncode != 0:
        raise RuntimeError(p_run.stderr[-3000:] if p_run.stderr else "ffmpeg export failed")

    print(f"Saved full-resolution high-quality video: {output_path.resolve()}")

def load_config_if_exists(config_path, default_p, frame_w, frame_h):
    path = Path(config_path)
    if not path.exists():
        print(f"No existing config found. Using default config: {path.resolve()}")
        return default_p

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        p = default_p.copy()

        for key in p.keys():
            if key in loaded:
                p[key] = loaded[key]

        p["left_x"] = clamp(int(p["left_x"]), 0, frame_w - 1)
        p["left_y"] = clamp(int(p["left_y"]), 0, frame_h - 1)
        p["left_w"] = clamp(int(p["left_w"]), 1, frame_w)
        p["left_h"] = clamp(int(p["left_h"]), 1, frame_h)

        p["right_x"] = clamp(int(p["right_x"]), 0, frame_w - 1)
        p["right_y"] = clamp(int(p["right_y"]), 0, frame_h - 1)
        p["right_w"] = clamp(int(p["right_w"]), 1, frame_w)
        p["right_h"] = clamp(int(p["right_h"]), 1, frame_h)

        p["right_rotate_180"] = bool(p["right_rotate_180"])
        p["gap_px"] = clamp(int(p["gap_px"]), 0, 200)
        p["preview_scale"] = clamp(int(p["preview_scale"]), 10, 100)

        print(f"Loaded existing config: {path.resolve()}")
        print(json.dumps(p, indent=2))
        return p

    except Exception as e:
        print(f"Failed to load config: {path.resolve()}")
        print(f"Reason: {e}")
        print("Using default config instead.")
        return default_p

def get_cropped_left_right(frame, p):
    h, w = frame.shape[:2]

    lx = clamp(int(p["left_x"]), 0, w - 1)
    ly = clamp(int(p["left_y"]), 0, h - 1)
    lw = clamp(int(p["left_w"]), 1, w - lx)
    lh = clamp(int(p["left_h"]), 1, h - ly)

    rx = clamp(int(p["right_x"]), 0, w - 1)
    ry = clamp(int(p["right_y"]), 0, h - 1)
    rw = clamp(int(p["right_w"]), 1, w - rx)
    rh = clamp(int(p["right_h"]), 1, h - ry)

    left = frame[ly:ly + lh, lx:lx + lw]
    right = frame[ry:ry + rh, rx:rx + rw]

    if p["right_rotate_180"]:
        right = cv2.flip(right, -1)

    target_h = min(left.shape[0], right.shape[0])
    left = left[:target_h, :]
    right = right[:target_h, :]

    return left, right

def build_preview(frame, p, apply_preview_scale=True):
    h, w = frame.shape[:2]

    lx = clamp(int(p["left_x"]), 0, w - 1)
    ly = clamp(int(p["left_y"]), 0, h - 1)
    lw = clamp(int(p["left_w"]), 1, w - lx)
    lh = clamp(int(p["left_h"]), 1, h - ly)

    rx = clamp(int(p["right_x"]), 0, w - 1)
    ry = clamp(int(p["right_y"]), 0, h - 1)
    rw = clamp(int(p["right_w"]), 1, w - rx)
    rh = clamp(int(p["right_h"]), 1, h - ry)

    left = frame[ly:ly + lh, lx:lx + lw]
    right = frame[ry:ry + rh, rx:rx + rw]

    if p["right_rotate_180"]:
        right = cv2.flip(right, -1)

    if p.get("seam_match_enable", False):
        gain = float(p.get("right_luma_gain", 1.0))
        gain = max(0.90, min(1.10, gain))
        right = apply_luma_gain_bgr(right, gain)


    target_h = min(left.shape[0], right.shape[0])
    left = left[:target_h, :]
    right = right[:target_h, :]

    if int(p["gap_px"]) > 0:
        gap = 255 * cv2.ones((target_h, int(p["gap_px"]), 3), dtype=frame.dtype)
        out = cv2.hconcat([left, gap, right])
    else:
        out = cv2.hconcat([left, right])

    if apply_preview_scale:
        scale = max(0.1, int(p["preview_scale"]) / 100.0)
        if scale != 1.0:
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return out


def save_processed_video(input_video, output_video, p, max_seconds=None):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds is not None and max_seconds > 0:
        max_frames = min(total_frames, int(max_seconds * fps))
    else:
        max_frames = total_frames

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame for output size detection.")

    first_out = build_preview(frame, p, apply_preview_scale=False)
    out_h, out_w = first_out.shape[:2]

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("avc1 failed, fallback to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    writer.write(first_out)

    frame_count = 1
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        out_frame = build_preview(frame, p, apply_preview_scale=False)

        if out_frame.shape[1] != out_w or out_frame.shape[0] != out_h:
            out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(out_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{max_frames} frames...")

    writer.release()
    cap.release()

    print(f"Saved full-resolution processed video: {output_path.resolve()}")
    print(f"Frames: {frame_count}, FPS: {fps:.3f}, Full output size: {out_w}x{out_h}")


def set_trackbar_values(win, p):
    cv2.setTrackbarPos("left_x", win, int(p["left_x"]))
    cv2.setTrackbarPos("left_y", win, int(p["left_y"]))
    cv2.setTrackbarPos("left_w", win, int(p["left_w"]))
    cv2.setTrackbarPos("left_h", win, int(p["left_h"]))

    cv2.setTrackbarPos("right_x", win, int(p["right_x"]))
    cv2.setTrackbarPos("right_y", win, int(p["right_y"]))
    cv2.setTrackbarPos("right_w", win, int(p["right_w"]))
    cv2.setTrackbarPos("right_h", win, int(p["right_h"]))

    cv2.setTrackbarPos("rotate180", win, 1 if p["right_rotate_180"] else 0)
    cv2.setTrackbarPos("gap_px", win, int(p["gap_px"]))
    cv2.setTrackbarPos("preview_%", win, int(p["preview_scale"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--frame", type=int, default=100, help="Frame index for calibration")
    parser.add_argument("--out", default="crop_calibration_space.json", help="Calibration JSON output path")
    parser.add_argument("--video-out", default=None, help="Processed preview video output path")
    parser.add_argument("--max-seconds", type=float, default=None, help="Limit saved video duration in seconds")
    parser.add_argument("--crf", type=int, default=0, help="FFmpeg CRF quality. Default 0 (lossless)")
    parser.add_argument("--preset", default="veryslow", help="FFmpeg preset. Default veryslow")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = clamp(args.frame, 0, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame index: {frame_idx}")

    h, w = frame.shape[:2]

    default_p = {
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
        "crf": args.crf,
        "preset": args.preset,
    }

    p = load_config_if_exists(args.out, default_p, w, h)

    win = "crop/hstack calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("left_x", win, 0, w - 1, lambda v: None)
    cv2.createTrackbar("left_y", win, 0, h - 1, lambda v: None)
    cv2.createTrackbar("left_w", win, 1, w, lambda v: None)
    cv2.createTrackbar("left_h", win, 1, h, lambda v: None)

    cv2.createTrackbar("right_x", win, 0, w - 1, lambda v: None)
    cv2.createTrackbar("right_y", win, 0, h - 1, lambda v: None)
    cv2.createTrackbar("right_w", win, 1, w, lambda v: None)
    cv2.createTrackbar("right_h", win, 1, h, lambda v: None)

    cv2.createTrackbar("rotate180", win, 1, 1, lambda v: None)
    cv2.createTrackbar("gap_px", win, 0, 200, lambda v: None)
    cv2.createTrackbar("preview_%", win, 50, 100, lambda v: None)

    set_trackbar_values(win, p)

    print("Controls:")
    print("  m = compute seam light match")
    print("  s = save calibration json")
    print("  v = export full-res high-quality video")
    print("  q / ESC = quit")

    while True:
        p["left_x"] = cv2.getTrackbarPos("left_x", win)
        p["left_y"] = cv2.getTrackbarPos("left_y", win)
        p["left_w"] = max(1, cv2.getTrackbarPos("left_w", win))
        p["left_h"] = max(1, cv2.getTrackbarPos("left_h", win))

        p["right_x"] = cv2.getTrackbarPos("right_x", win)
        p["right_y"] = cv2.getTrackbarPos("right_y", win)
        p["right_w"] = max(1, cv2.getTrackbarPos("right_w", win))
        p["right_h"] = max(1, cv2.getTrackbarPos("right_h", win))

        p["right_rotate_180"] = bool(cv2.getTrackbarPos("rotate180", win))
        p["gap_px"] = cv2.getTrackbarPos("gap_px", win)
        p["preview_scale"] = max(10, cv2.getTrackbarPos("preview_%", win))

        preview = build_preview(frame, p, apply_preview_scale=True)
        cv2.imshow(win, preview)

        key = cv2.waitKey(30) & 0xFF

        if key in [ord("q"), 27]:
            break

        if key == ord("s"):
            out_path = Path(args.out)
            out_path.write_text(json.dumps(p, indent=2), encoding="utf-8")
            print(f"Saved calibration JSON: {out_path.resolve()}")
            print(json.dumps(p, indent=2))

        if key == ord("v"):
            if args.video_out is None:
                input_path = Path(args.video)
                video_out = input_path.with_name(f"{input_path.stem}_cropcalib_fullres_hq.mp4")
            else:
                video_out = Path(args.video_out)

            save_processed_video_ffmpeg(
                input_video=args.video,
                output_video=video_out,
                p=p,
                max_seconds=args.max_seconds,
                crf=args.crf,
                preset=args.preset,
            )
        if key == ord("m"):
            left, right = get_cropped_left_right(frame, p)

            gain, left_y, right_y = compute_seam_luma_gain(
                left,
                right,
                seam_width_px=p.get("seam_width_px", 30),
                min_gain=0.90,
                max_gain=1.10,
            )

            p["seam_match_enable"] = True
            p["right_luma_gain"] = float(gain)

            print(f"Matched seam:")
            print(f"  left_y  = {left_y:.2f}")
            print(f"  right_y = {right_y:.2f}")
            print(f"  gain    = {gain:.4f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
