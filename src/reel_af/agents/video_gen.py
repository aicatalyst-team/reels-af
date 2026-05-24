"""Video generation — grok-imagine first frame → Veo image-to-video.

For each shot:
  1. Generate a vertical first frame with grok-imagine (fast, ~5s, $0.03).
  2. Feed it to Veo 3.1 Lite as image_url + first_frame, plus the motion
     prompt from the shot director. Veo produces a 4-second 720×1280 MP4
     with the requested motion starting from that frame.

This gives us per-shot motion AND visual consistency across shots (because
each shot has its own coherent first-frame). The grok-imagine pass is the
trust anchor; Veo just animates it.

Cost: ~$0.32/Veo + ~$0.03/grok-imagine = ~$0.35 per shot. 5 shots = ~$1.75/reel.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from agentfield.media_providers import OpenRouterProvider

from reel_af.agents.scene_breaker import Scene
from reel_af.agents.shot_director_v2 import ShotPlanV2
from reel_af.models import BeatArtifact

IMAGE_MODEL = "openrouter/x-ai/grok-imagine-image-quality"
VIDEO_MODEL = "openrouter/google/veo-3.1-lite"

# Veo accepts duration 4 / 6 / 8 seconds. We clamp segment durations to the
# nearest accepted value, slightly OVER the spoken duration so the video has
# a tail; ffmpeg will trim to the actual audio length at assembly.
_VEO_DURATIONS = (4, 6, 8)

# Global style block appended to every image prompt so all shots feel like
# the same reel. Editable in one place.
STYLE_NOTE = (
    "cinematic documentary still, warm natural light, shallow depth of field, "
    "35mm film grain, vertical 9:16 framing, fills the frame, no text or letters"
)


def _veo_duration(est_s: float) -> int:
    """Pick the smallest accepted Veo duration ≥ est_s, capped at 8s."""
    for d in _VEO_DURATIONS:
        if d >= est_s:
            return d
    return _VEO_DURATIONS[-1]


def _augment(prompt: str) -> str:
    """Append the global style block to an image prompt."""
    base = prompt.strip().rstrip(".")
    return f"{base}. {STYLE_NOTE}."


def _image_to_data_url(path: Path) -> str:
    """Encode a local JPEG/PNG as a data: URL so Veo can use it as first_frame."""
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{base64.b64encode(path.read_bytes()).decode()}"


async def _gen_first_frame(
    provider: OpenRouterProvider,
    plan: ShotPlanV2,
    idx: int,
    out_dir: Path,
) -> Path:
    prompt = _augment(plan.image_prompt)
    result = await provider.generate_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        image_config={"aspect_ratio": "9:16"},
    )
    if not result.images:
        raise RuntimeError(f"video_gen: grok-imagine returned no first frame for shot {idx}")
    out = out_dir / f"seg-{idx:02d}-frame.jpg"
    result.images[0].save(str(out))
    return out


async def _gen_video(
    provider: OpenRouterProvider,
    plan: ShotPlanV2,
    seg: Scene,
    first_frame: Path,
    out_dir: Path,
) -> Path:
    """Veo image-to-video. Uses the grok-imagine still as the starting frame."""
    frame_url = _image_to_data_url(first_frame)
    duration = _veo_duration(seg.est_duration_s)
    # Compose Veo prompt: the literal scene is set by first_frame; the motion
    # prompt is what should HAPPEN. We also pass the on-screen-text context
    # so Veo doesn't try to ALSO put text in the video (it sometimes does).
    veo_prompt = (
        f"{plan.motion_prompt}. {STYLE_NOTE}. "
        f"Do not add any text, captions, or letters to the frame."
    )

    result = await provider.generate_video(
        prompt=veo_prompt,
        model=VIDEO_MODEL,
        duration=duration,
        aspect_ratio="9:16",
        resolution="720p",
        # Use frame_images with frame_type=first_frame for proper i2v anchoring.
        frame_images=[
            {"type": "image_url", "image_url": {"url": frame_url}, "frame_type": "first_frame"}
        ],
        # Generous timeout — Veo Lite takes ~30-90s end-to-end.
        poll_interval=8.0,
        timeout=420.0,
    )
    if not result.videos:
        raise RuntimeError(f"video_gen: Veo returned no video for shot {seg.idx}")
    out = out_dir / f"seg-{seg.idx:02d}.mp4"
    result.videos[0].save(str(out))
    return out


async def _still_as_video(
    still_path: Path, duration: float, out_path: Path
) -> Path:
    """Fallback when Veo moderates / fails — render the still as a 4s MP4
    with a slow ken-burns zoom so the scene still has motion."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", "30", "-t", f"{duration:.3f}",
        "-i", str(still_path),
        "-vf",
        f"scale=1280:2280:force_original_aspect_ratio=increase,"
        f"crop=1280:2280,"
        f"crop=720:1280:"
        f"x='(100 - 100*t/{max(duration, 0.1):.3f})':"
        f"y='(500 - 500*t/{max(duration, 0.1):.3f})',"
        f"fps=30,format=yuv420p",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "21",
        "-r", "30",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"video_gen fallback ffmpeg failed: {err.decode(errors='replace')[-400:]}"
        )
    return out_path


async def gen_shot(
    provider: OpenRouterProvider,
    seg: Scene,
    plan: ShotPlanV2,
    out_dir: Path,
) -> BeatArtifact:
    """Generate first-frame + video for one segment, with still fallback.

    If Veo fails (most often: content moderation false-positive), don't
    take down the whole reel — render the still as a slow-zoom 4s MP4 so
    the scene still has motion and the timing stays intact.
    """
    frame = await _gen_first_frame(provider, plan, seg.idx, out_dir)
    try:
        video = await _gen_video(provider, plan, seg, frame, out_dir)
    except Exception as e:
        print(f"[video_gen] scene {seg.idx} Veo failed ({e}); falling back to still.")
        fallback = out_dir / f"seg-{seg.idx:02d}-fallback.mp4"
        video = await _still_as_video(frame, duration=4.0, out_path=fallback)
    # We hijack `image_path` to carry the first frame for fallback rendering
    # and add the .mp4 path later via assembly.
    return BeatArtifact(idx=seg.idx, image_path=video)  # video stored in image_path slot


async def generate_videos(
    segments: list[Scene],
    plans: list[ShotPlanV2],
    out_dir: Path,
) -> list[BeatArtifact]:
    """Fan-out video generation across all segments."""
    if len(segments) != len(plans):
        raise ValueError("video_gen: segments and plans length mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = OpenRouterProvider()
    results = await asyncio.gather(
        *(gen_shot(provider, s, p, out_dir) for s, p in zip(segments, plans)),
        return_exceptions=True,
    )
    errs = [r for r in results if isinstance(r, Exception)]
    if errs:
        raise RuntimeError(
            f"video_gen: {len(errs)}/{len(segments)} shots failed. First error: {errs[0]}"
        )
    return [r for r in results if isinstance(r, BeatArtifact)]
