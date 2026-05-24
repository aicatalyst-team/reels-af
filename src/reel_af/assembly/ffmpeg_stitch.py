"""ffmpeg-based stitcher: per-beat clips → concatenated vertical MP4.

Each beat becomes one 1080×1920 H.264 clip with:
  • the generated image, scaled to fill, with a slow ken-burns motion
  • the burned-in caption text (centered, big sans-serif, white w/ black outline)
  • the per-beat VO overlaid as audio

The final reel is built with ffmpeg's concat demuxer (no re-encoding of the
already-encoded beats — fast and lossless).

ffmpeg quirks worth knowing:
  • zoompan operates in z-units (1.0 = no zoom). Long durations + steep
    z-gain produce noticeable stepping. We keep gain mild (~5–8%).
  • drawtext needs a font path. We pick the first font that exists on the
    host so the example works on macOS and Linux without setup.
  • libx264 + yuv420p is what TikTok / Reels / Shorts accept directly.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
from pathlib import Path

from reel_af.models import Beat, BeatArtifact, ReelResult, Storyboard

# Vertical 9:16 — Instagram Reels / TikTok / YouTube Shorts native size.
TARGET_W = 1080
TARGET_H = 1920
FPS = 60  # 60 fps eliminates the temporal stepping zoompan was producing.

# Pre-scaled source size — gives the crop window ~12% room to move without
# revealing edges. Linear crop animation in this resolution is smooth (no
# float-quantization artifacts the way zoompan has).
SRC_W = int(TARGET_W * 1.12)  # 1209
SRC_H = int(TARGET_H * 1.12)  # 2150
SLACK_W = SRC_W - TARGET_W    # 129
SLACK_H = SRC_H - TARGET_H    # 230

# Candidate font paths in order of preference. First hit wins.
_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _find_font() -> str:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    raise RuntimeError(
        "ffmpeg-stitch: no font found. Install one of: "
        + ", ".join(_FONT_CANDIDATES)
    )


def _ffmpeg_escape(s: str) -> str:
    """Escape a string for use inside an ffmpeg drawtext text= argument."""
    return (
        s.replace("\\", "\\\\")
         .replace(":", r"\:")
         .replace("'", r"\'")
         .replace(",", r"\,")
    )


def _motion_crop_xy(motion: str, duration_s: float) -> tuple[str, str]:
    """Return ffmpeg crop x= and y= expressions for the chosen motion.

    Crop animates linearly across the SRC_W/SRC_H slack window, producing
    rock-steady motion at any framerate (no zoompan quantization).
    """
    # Normalised time over the clip, clamped to [0, 1].
    p = f"min(t/{max(duration_s, 0.05):.6f},1)"
    cx = SLACK_W // 2
    cy = SLACK_H // 2
    if motion == "zoom_in":
        # Implemented as a slow centering crop — visually feels like zoom-in
        # because the crop window shrinks toward center. Cheaper than scale().
        return f"{cx} - {cx} * {p} * 0.4", f"{cy} - {cy} * {p} * 0.4"
    if motion == "zoom_out":
        return f"{cx} * {p} * 0.4", f"{cy} * {p} * 0.4"
    if motion == "pan_left":
        return f"{SLACK_W} - {SLACK_W} * {p}", str(cy)
    if motion == "pan_right":
        return f"{SLACK_W} * {p}", str(cy)
    # static — fixed crop at center
    return str(cx), str(cy)


async def _render_beat(
    beat: Beat,
    image_path: Path,
    audio_path: Path,
    out_path: Path,
    font_path: str,
) -> None:
    """Render one beat to a self-contained MP4 with motion, caption, and VO.

    Pipeline:
      1. Scale image to SRC_W × SRC_H (covering).
      2. Animated crop window (SLACK_W/SLACK_H of room) → buttery-smooth motion.
      3. Caption: ALL-CAPS, big sans-serif, white, sitting on a semi-opaque
         dark pill in the lower third. Reads in 0.3s with sound off.
    """
    caption_text = beat.caption.upper().strip()
    caption_esc = _ffmpeg_escape(caption_text)
    cx, cy = _motion_crop_xy(beat.motion_hint, beat.duration_s)

    # Sizes tuned for 1080×1920 vertical: cap fontsize ~115pt is big enough
    # to read at thumbnail size but leaves headroom for 4-word captions.
    fontsize = 115 if len(caption_text) <= 16 else 92

    drawtext = (
        f"drawtext=fontfile={font_path}"
        f":text='{caption_esc}'"
        f":fontsize={fontsize}:fontcolor=white"
        # Black pill behind the text gives instant readability over any image.
        f":box=1:boxcolor=black@0.75:boxborderw=28"
        # Subtle white outline keeps text crisp on dark/light hybrid frames.
        f":bordercolor=white@0.0:borderw=0"
        f":x=(w-text_w)/2"
        # Lower third (~70% down) sits above the IG/TikTok bottom UI overlay.
        f":y=h*0.70"
    )

    vfilter = (
        # Fit, cover, then animate.
        f"scale={SRC_W}:{SRC_H}:force_original_aspect_ratio=increase,"
        f"crop={SRC_W}:{SRC_H},"
        f"crop={TARGET_W}:{TARGET_H}:x='{cx}':y='{cy}',"
        # Lock the output framerate AFTER motion so playback is uniform.
        f"fps={FPS},"
        f"format=yuv420p,"
        + drawtext
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-t", f"{beat.duration_s:.3f}",
        "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", vfilter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "19",
        "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg beat {beat.idx} failed (exit {proc.returncode}):\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  stderr: {stderr.decode(errors='replace')[-800:]}"
        )


async def _concat_clips(clip_paths: list[Path], out_path: Path) -> None:
    """Concat per-beat clips losslessly via the concat demuxer."""
    list_file = out_path.parent / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed (exit {proc.returncode}):\n"
            f"  stderr: {stderr.decode(errors='replace')[-800:]}"
        )


def _index_artifacts(artifacts: list[BeatArtifact]) -> dict[int, BeatArtifact]:
    """Merge image+audio artifacts (which arrive separately) into one map per beat."""
    merged: dict[int, BeatArtifact] = {}
    for a in artifacts:
        cur = merged.get(a.idx)
        if cur is None:
            merged[a.idx] = a
        else:
            merged[a.idx] = BeatArtifact(
                idx=a.idx,
                image_path=cur.image_path or a.image_path,
                audio_path=cur.audio_path or a.audio_path,
            )
    return merged


async def stitch(
    storyboard: Storyboard,
    image_artifacts: list[BeatArtifact],
    audio_artifacts: list[BeatArtifact],
    out_dir: Path,
    run_id: str,
) -> ReelResult:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install with `brew install ffmpeg`.")
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_font()

    merged = _index_artifacts(image_artifacts + audio_artifacts)

    clip_paths: list[Path] = []
    for beat in storyboard.beats:
        art = merged.get(beat.idx)
        if not art or not art.image_path or not art.audio_path:
            raise RuntimeError(f"stitch: missing image or audio for beat {beat.idx}")
        clip = out_dir / f"clip-{beat.idx:02d}.mp4"
        await _render_beat(
            beat=beat,
            image_path=art.image_path,
            audio_path=art.audio_path,
            out_path=clip,
            font_path=font_path,
        )
        clip_paths.append(clip)

    final = out_dir / "reel.mp4"
    await _concat_clips(clip_paths, final)

    return ReelResult(
        output_path=final,
        storyboard=storyboard,
        duration_s=storyboard.total_duration_s,
        suggested_caption=f"{storyboard.angle.hook_line} {storyboard.angle.angle}",
        suggested_hashtags=[],  # filled by a later agent
        run_id=run_id,
    )
