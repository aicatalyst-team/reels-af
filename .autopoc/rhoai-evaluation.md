# RHOAI Fitness Evaluation: reels-af

## Project Summary
AI-native viral short-form video producer built on the AgentField multi-agent framework. Takes a URL or topic phrase and produces 1080x1920 vertical reels with word-burst karaoke subtitles in ~80 seconds, orchestrating 18 specialized reasoners through a DAG.

## Impact Dimensions

| Dimension | Score (0-20) | Rationale |
|-----------|-------------|-----------|
| Audience Value | 14 | Strong developer audience appeal - video generation from AI agents is highly topical. Multi-agent DAG architecture is interesting for platform teams. |
| Strategic Alignment | 12 | Demonstrates agentic AI orchestration patterns relevant to GenAI Studio. Uses LLM APIs for reasoning, TTS, and image generation. |
| Strategy Fit | 12 | Fits agentic-ai category. Multi-agent workflow via control-plane + worker nodes aligns with agent-runtime capability. |
| Platform Leverage | 14 | Two-service architecture (control-plane + worker) maps naturally to K8s. The AgentField UI at :8080/ui provides live DAG visualization. PVC for output artifacts, health endpoint, async API are cloud-native patterns. |
| Demo Potential | 16 | Excellent demo potential - produces actual video files. Live DAG visualization in UI. Simple curl to trigger, visual output in ~80s. |

**Impact Score**: (14 + 12 + 12 + 14 + 16) / 5 = **13.6 / 20**

## Feasibility Dimensions

| Dimension | Score (0-20) | Rationale |
|-----------|-------------|-----------|
| Container Readiness | 16 | Has working Dockerfile and docker-compose.yml. Needs UBI conversion but structure is clean. |
| Dependency Profile | 14 | Standard Python packages (pydantic, aiohttp, Pillow). agentfield SDK is the wildcard. ffmpeg needed as system dep. |
| Reproduction Confidence | 10 | Requires running AgentField control-plane sidecar + OPENROUTER_API_KEY with credits. SDK patches suggest some instability. |
| Complexity Sweet Spot | 14 | Good balance - 25 Python files with clear structure. Multi-service requirement adds operational complexity but is manageable. |

**Feasibility Score**: (16 + 14 + 10 + 14) / 4 = **13.5 / 20**

## Overall Assessment

- **Relationship**: adjacent (agentic AI tooling, multi-agent orchestration)
- **Strategy Areas**: agentic-ai, agent-runtime
- **Capability Labels**: llama-stack, mcp, agent-runtime, ai-hub, genai-studio
- **Strengths**: Working Dockerfile, compelling visual output, cloud-native architecture
- **Risks**: External API dependency (OpenRouter), third-party control-plane image compatibility, SDK patches
- **License**: Apache-2.0 (enterprise compatible)

## Recommendation
**PROCEED** - Strong demo potential and natural K8s fit outweigh the API dependency risk. The OPENROUTER_API_KEY can be provided via K8s Secret. The two-service architecture demonstrates real platform value.
