"""Assembly v2 — concat per-segment Veo clips + clean capcut-style captions.

For each segment:
  1. Take the Veo MP4 (~720p 9:16, 4-8s) and the segment's TTS .wav.
  2. Scale Veo output up to 1080×1920 (Reels/TikTok native size).
  3. Trim the video to exactly the spoken duration (with a small head/tail
     fade so cuts don't jar).
  4. Overlay the on_screen_text as a centered phrase in the bottom-third
     with drop-shadow + quick fade-in (no heavy boxes — that was too "ad-y").
  5. Replace Veo's silent / sometimes-music audio with the Kokoro VO only.

Then concat all segment clips. Final reel is 1080×1920 H.264 + AAC, ready
to upload to TikTok / Reels / Shorts without re-encoding.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
from pathlib import Path

from reel_af.agents.scene_breaker import Scene
from reel_af.agents.shot_director_v2 import ShotPlanV2
from reel_af.assembly.caption_fit import fit_caption
from reel_af.models import BeatArtifact, ReelResult, Storyboard

# Hold the last scene's final frame for this long after the VO ends so the
# reel doesn't cut abruptly. Quick enough to keep retention; long enough to
# let the close land + give the viewer a save-window.
FINAL_HOLD_S = 1.5
# Brief silence + frozen first frame at the very start. Gives viewers a
# half-beat to register the hook visual before the voice starts — almost
# every viral reel does this.
START_PAD_S = 0.4

TARGET_W = 1080
TARGET_H = 1920
FPS = 30  # Veo outputs 24-30fps; matching avoids interpolation artefacts.

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
    raise RuntimeError("assembly: no usable font found.")


def _ffmpeg_escape(s: str) -> str:
    # Replace ASCII apostrophe with typographic ’ — sidesteps ffmpeg
    # drawtext's broken handling of single quotes inside text='...'
    # (we'd otherwise need shell-style break-quote-rejoin sequences).
    s = s.replace("'", "’")
    # Same for straight double-quote.
    s = s.replace('"', "”")
    # The ASCII percent sign is the start of drawtext's `%{...}` placeholder
    # syntax. Even with backslash escape (`\%`) drawtext silently drops the
    # ENTIRE text — so a caption "30%" renders as a blank frame. Helvetica
    # (our default font) also lacks the fullwidth `％`, so we can't swap
    # glyphs. Substitute the spelled-out " PCT" — readable, unambiguous,
    # and unaffected by drawtext's placeholder parser.
    s = s.replace("%", " PCT")
    return (
        s.replace("\\", "\\\\")
         .replace(":", r"\:")
         .replace(",", r"\,")
    )


def _probe_duration(path: Path) -> float:
    """Get the duration in seconds of a media file via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


async def _render_segment(
    seg: Scene,
    plan: ShotPlanV2,
    video_path: Path,      # Veo clip
    audio_path: Path,      # Kokoro VO
    out_path: Path,
    font_path: str,
    is_final: bool = False,
    is_first: bool = False,
) -> None:
    audio_dur = _probe_duration(audio_path)
    # Target clip duration = spoken duration + small tail. The final scene
    # gets a longer hold so the close doesn't cut abruptly.
    tail = FINAL_HOLD_S if is_final else 0.15
    # NO head-pad on first scene. Previously we froze the first frame for
    # 0.4s with tpad — but tpad's PTS rewrite broke drawtext's alpha
    # animation, so clip 0 rendered captionless. Voice + visual + caption
    # all start together; the viewer registers them simultaneously.
    head = 0.0
    clip_dur = head + audio_dur + tail

    # Caption: measured-width fit. Picks the largest single-line fontsize
    # that fits the safe area; falls back to 2-line wrap if even small text
    # would overflow. No more clipped text.
    text = seg.caption.upper().strip()
    layout = fit_caption(text, font_path)

    # Capcut-style pop-in: caption appears at 0.08s with a 0.18s fade.
    fade_in_start = 0.08
    fade_in_end = 0.26
    alpha_expr = (
        f"if(lt(t,{fade_in_start}),0,"
        f"if(lt(t,{fade_in_end}),(t-{fade_in_start})/{fade_in_end-fade_in_start},1))"
    )

    # Caption position: lower-third (~72% down) clears platform UI.
    # For 2-line layouts, stack: first line at 70%, second at 76%.
    line_height_frac = 0.06  # ~115px between lines at 1080×1920
    y_anchor = 0.72 - (len(layout.lines) - 1) * line_height_frac / 2

    drawtext_chain: list[str] = []
    for i, line in enumerate(layout.lines):
        line_esc = _ffmpeg_escape(line)
        y_pos = f"h*{y_anchor + i * line_height_frac}"
        drawtext_chain.append(
            f"drawtext=fontfile={font_path}"
            f":text='{line_esc}'"
            f":fontsize={layout.fontsize}:fontcolor=white"
            f":x=(w-text_w)/2:y={y_pos}"
            f":bordercolor=black@0.9:borderw=4"
            f":shadowcolor=black@0.7:shadowx=4:shadowy=5"
            f":alpha='{alpha_expr}'"
        )
    drawtext_main = ",".join(drawtext_chain)

    # Filter chain: scale Veo output → optional start/end frame-holds → trim.
    # tpad=start_mode=clone freezes the FIRST frame at the head;
    # tpad=stop_mode=clone freezes the LAST frame at the tail.
    # Used together for the first scene (head pad) and last scene (tail pad).
    pad_v = ""
    if head > 0:
        pad_v += f"tpad=start_mode=clone:start_duration={head:.3f},"
    if is_final:
        pad_v += f"tpad=stop_mode=clone:stop_duration={tail + 0.5:.3f},"

    vfilter = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},"
        + pad_v
        + f"trim=end={clip_dur:.3f},setpts=PTS-STARTPTS,"
        f"fps={FPS},"
        f"format=yuv420p,"
        + drawtext_main
    )

    # Audio: adelay shifts the VO by `head` ms (silence at start);
    # apad pads silence at end for the final hold; trim + tiny fade-out.
    head_ms = int(head * 1000)
    afilter_parts = []
    if head_ms > 0:
        # adelay needs a delay per channel; "Nms|Nms" covers mono and stereo.
        afilter_parts.append(f"adelay={head_ms}|{head_ms}")
    if is_final and tail > 0:
        afilter_parts.append(f"apad=pad_dur={tail:.3f}")
    afilter_parts.append(f"atrim=end={clip_dur:.3f}")
    afilter_parts.append("asetpts=PTS-STARTPTS")
    afilter_parts.append(
        f"afade=t=out:st={max(head + audio_dur - 0.05, 0):.3f}:d=0.10"
    )
    afilter = ",".join(afilter_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex",
        f"[0:v]{vfilter}[v];[1:a]{afilter}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "19",
        "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-movflags", "+faststart",
        "-t", f"{clip_dur:.3f}",
        str(out_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg segment {seg.idx} failed (exit {proc.returncode}):\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  stderr: {stderr.decode(errors='replace')[-800:]}"
        )


async def _concat_clips(clip_paths: list[Path], out_path: Path) -> None:
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
        "-movflags", "+faststart",
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


async def stitch_v2(
    segments: list[Scene],
    plans: list[ShotPlanV2],
    video_artifacts: list[BeatArtifact],   # image_path slot holds the .mp4
    audio_artifacts: list[BeatArtifact],
    out_dir: Path,
    run_id: str,
    storyboard_for_result: Storyboard,     # passed through for ReelResult
) -> ReelResult:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("assembly: ffmpeg / ffprobe not found. `brew install ffmpeg`.")
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = _find_font()

    # Index audio and video artifacts by segment idx.
    audio_by_idx = {a.idx: a.audio_path for a in audio_artifacts if a.audio_path}
    video_by_idx = {v.idx: v.image_path for v in video_artifacts if v.image_path}

    # Build per-segment render tasks and run them in PARALLEL — each ffmpeg
    # subprocess saturates ~1 core; on a multi-core box they overlap cleanly
    # and the wall time collapses to ~max(segment) instead of sum(segments).
    render_jobs = []
    clip_paths: list[Path] = []
    last_idx = len(segments) - 1
    for i, (seg, plan) in enumerate(zip(segments, plans)):
        vid = video_by_idx.get(seg.idx)
        aud = audio_by_idx.get(seg.idx)
        if not vid or not aud:
            raise RuntimeError(f"assembly: missing video or audio for segment {seg.idx}")
        out_clip = out_dir / f"clip-{seg.idx:02d}.mp4"
        clip_paths.append(out_clip)
        render_jobs.append(
            _render_segment(
                seg, plan, vid, aud, out_clip, font_path,
                is_final=(i == last_idx),
                is_first=(i == 0),
            )
        )
    await asyncio.gather(*render_jobs)

    total_dur = sum(_probe_duration(p) for p in clip_paths)
    final = out_dir / "reel.mp4"
    await _concat_clips(clip_paths, final)

    return ReelResult(
        output_path=final,
        storyboard=storyboard_for_result,
        duration_s=total_dur,
        suggested_caption="",
        suggested_hashtags=[],
        run_id=run_id,
    )
