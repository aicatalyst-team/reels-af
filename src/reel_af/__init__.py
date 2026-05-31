"""reel-af — turn a URL or a topic into a vertical viral reel.

Two entry points (also exposed as DAG-visible reasoners on the
AgentField control plane):

    from reel_af.app import article_to_reel, topic_to_reel

    await article_to_reel("https://example.com/article")
    await topic_to_reel("philosophy of mind")

The pipeline is OpenRouter-only: Gemini 3.1 Flash TTS, Gemini 2.5
Flash Image, Veo 3.1 Lite i2v. Subtitles via libass; final stitch is a
single sample-accurate ffmpeg pass.
"""

__version__ = "1.0.0"
