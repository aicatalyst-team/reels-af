"""Media rendering — TTS, images, video, subtitles, stitch.

All paths use OpenRouter exclusively:
  tts       — Gemini 3.1 Flash TTS via /audio/speech (sentence-by-sentence)
  images    — Gemini 2.5 Flash Image (first frame per beat)
  video     — Veo 3.1 Lite image-to-video (one clip per beat)
  subtitles — pysubs2 + libass (word-burst karaoke + optional accent)
  stitch    — single ffmpeg pass: concat + libass + audio mux
"""
