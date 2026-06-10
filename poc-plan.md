# PoC Plan: reels-af

## Project Overview
reels-af is an AI-native viral short-form video producer built on the AgentField multi-agent framework. It converts URLs or topic phrases into 1080x1920 vertical reels with word-burst karaoke subtitles in approximately 80 seconds.

## PoC Type
**llm-app** — Multi-agent AI application using external LLM APIs for reasoning, TTS, image generation, and video rendering.

## Architecture

The application consists of two services:
1. **control-plane**: AgentField control plane (Go binary, pre-built image `agentfield/control-plane:latest`) — orchestrates the DAG of 18 specialized reasoners
2. **reel-af**: Python worker node — registers with the control plane, serves reasoners on port 8002, performs ffmpeg-based video rendering

## Infrastructure Requirements

| Requirement | Value |
|------------|-------|
| GPU | No |
| PVC | Yes (output artifacts) |
| Sidecars | No (control-plane is a separate Deployment) |
| Resource Profile | medium |
| Deployment Model | deployment |
| Listens on Port | Yes (control-plane: 8080, reel-af: 8002) |
| Long Running | Yes |
| Needs LLM API | Yes (OpenRouter for reasoning, TTS, image gen) |
| LLM Env Pattern | custom (OPENROUTER_API_KEY) |

## PoC Components
1. **control-plane** — Use pre-built image `agentfield/control-plane:latest`
2. **reel-af** — Build from Dockerfile, converted to UBI base

## Test Scenarios

### Scenario 1: Health Check
- **Type**: http
- **Description**: Verify the reel-af worker starts and responds on its health endpoint
- **Endpoint**: /health
- **Expected**: HTTP 200 response
- **Timeout**: 30s

### Scenario 2: Control Plane Reachability
- **Type**: http
- **Description**: Verify the AgentField control plane UI is accessible
- **Endpoint**: / (on control-plane service port 8080)
- **Expected**: HTTP 200 response with UI content
- **Timeout**: 30s

### Scenario 3: Agent Registration
- **Type**: http
- **Description**: Verify reel-af registers as a node with the control plane
- **Endpoint**: /api/v1/nodes (on control-plane)
- **Expected**: JSON response listing the reel-af node
- **Timeout**: 60s

### Scenario 4: Pipeline Invocation (Dry Run)
- **Type**: http
- **Description**: Attempt to trigger the article-to-reel pipeline. Without OPENROUTER_API_KEY, this should return an appropriate error indicating the API key is required.
- **Endpoint**: /api/v1/execute/async/reel-af.reel_article_to_reel
- **Input**: {"input": {"url": "https://example.com/test-article"}}
- **Expected**: Either execution starts (if key present) or returns error about missing API key
- **Timeout**: 30s

## Environment Variables

| Variable | Source | Required |
|----------|--------|----------|
| OPENROUTER_API_KEY | Secret | Yes (for reel generation) |
| AGENTFIELD_SERVER | ConfigMap | Yes (http://control-plane:8080) |
| AGENT_CALLBACK_URL | ConfigMap | Yes (http://reel-af:8002) |
| AGENT_NODE_ID | ConfigMap | Yes (reel-af) |
| REEL_AF_MODEL | ConfigMap | No (defaults to openrouter/deepseek/deepseek-v4-pro) |
| PORT | ConfigMap | No (defaults to 8002) |

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| OPENROUTER_API_KEY not available | Deploy infrastructure, validate health and registration; document key requirement |
| agentfield/control-plane image not OCI-compatible | Test pull and run; fall back to health-only validation if control-plane fails |
| SDK patches indicate instability | Monitor pod logs for SDK-related errors |
| Large image size (ffmpeg + fonts) | Use multi-stage or minimal system deps |
