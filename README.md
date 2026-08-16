# Polymodel MCP Server

One MCP server for external text and image models across OpenAI, Google Gemini, and Together AI. Built on the MCP TypeScript SDK v2 (2026-07-28 spec); it supersedes the separate `openai-mcp` and `gemini-mcp` servers.

## Models

| Alias | Modality | Underlying model | Provider |
|---|---|---|---|
| `gpt` | text | gpt-5.6-sol | OpenAI |
| `gemini` | text | gemini-pro-latest | Google |
| `kimi-k3` | text | moonshotai/Kimi-K3 | Together AI |
| `gpt-image` | image | gpt-image-2 | OpenAI |
| `nano-banana` | image | gemini-3.1-flash-image | Google |
| `flux-2-dev` | image | black-forest-labs/FLUX.2-dev | Together AI |

## Setup

```sh
cp .env.example .env   # add the keys you have; models without a key are not registered
make build
```

Register with Claude Code:

```sh
claude mcp add polymodel -- node /path/to/polymodel/dist/index.js
```

## Tools

- `generate_text(model, prompt, max_tokens?, temperature?, reasoning_effort?)`
- `generate_image(model, prompt, width?, height?, output_path?, return_image?)` — defaults to 750x750; each provider snaps to its nearest supported size and reports the adjustment. The image is saved to disk and the path returned.
- `list_models()` — registered models with capabilities.
- `get_version()`

## Adding a model

Implement `TextModel` or `ImageModel` from `src/domain/types.ts` and register it in `src/index.ts`. The tool layer only depends on the `ModelRegistry`, so no other changes are needed.
