# Polymodel MCP Server

One MCP server for external text, image, sound effect and music models across OpenAI, Google Gemini, Together AI and ElevenLabs. Built on the MCP TypeScript SDK v2 (2026-07-28 spec); it supersedes the separate `openai-mcp` and `gemini-mcp` servers.

## Models

| Alias | Modality | Underlying model | Provider |
|---|---|---|---|
| `gpt` | text | gpt-6-astra | OpenAI |
| `gemini` | text | gemini-pro-latest (currently gemini-3.1-pro-preview) | Google |
| `kimi-k3` | text | moonshotai/Kimi-K3 | Together AI |
| `gpt-image` | image | gpt-image-2.5-sunburst | OpenAI |
| `nano-banana` | image | gemini-3-pro-image | Google |
| `flux-2-max` | image | black-forest-labs/FLUX.2-max | Together AI |
| `eleven-sfx` | sound | eleven_text_to_sound_v2 | ElevenLabs |
| `eleven-music` | music | music_v2_5 | ElevenLabs |
| `eleven-music-v2` | music | music_v2 | ElevenLabs |
| `eleven-music-v1` | music | music_v1 (deprecated by ElevenLabs) | ElevenLabs |

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

- `generate_text(model, prompt, max_tokens?, temperature?, reasoning_effort?)` — `max_tokens` is the answer budget. Kimi K3 and Gemini both always reason and charge that thinking against the same limit, so those providers add a reasoning allowance on top rather than letting thinking eat the answer. Each provider also maps `reasoning_effort` onto the levels it accepts.
- `generate_image(model, prompt, width?, height?, output_path?, return_image?)` — defaults to 750x750; each provider snaps to its nearest supported size and reports the adjustment. The image is saved to disk and the path returned.
- `generate_sound_effect(prompt, model?, duration_seconds?, loop?, prompt_influence?, output_format?, output_path?, return_audio?)`
- `generate_music(prompt | composition_plan, model?, ...)` — covers every Music API option, including generation mode, finetunes, inpainting storage, lyric timestamps and C2PA signing. Saves the audio plus a `.json` file holding the composition plan the model used, so the song can be edited later.
- `create_music_plan(prompt, model?, music_length_ms?, source_composition_plan?)` — draft or revise a plan without generating audio.
- `generate_music_from_video(video_paths, model?, description?, tags?, ...)`
- `upload_music(file_path, ...)` — store your own audio for inpainting and optionally extract its plan.
- `separate_stems(file_path, stem_variation?, ...)` — two or six stems, unpacked into a directory.
- `manage_music_finetunes(action, ...)` — list, get, create, update or delete finetunes.
- `list_models()` — registered models with capabilities.
- `get_version()`

Tools for a modality appear only when a key for one of its providers is set.

## Adding a model

Implement `TextModel`, `ImageModel`, `SoundModel` or `MusicModel` from `src/domain/types.ts` and register it in `src/index.ts`. The tool layer only depends on the `ModelRegistry`, so no other changes are needed.
