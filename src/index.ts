#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/server';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import * as z from 'zod';
import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { basename, join, resolve } from 'path';
import { sniffImage } from './domain/imageInfo.js';
import { readZip } from './domain/zip.js';
import { AudioFile, InputImage, ModelInfo } from './domain/types.js';
import { createRequire } from 'module';
import { loadConfig } from './config.js';
import { ModelRegistry } from './registry.js';
import { createOpenAIModels } from './providers/openai.js';
import { createGeminiModels } from './providers/gemini.js';
import { createTogetherModels } from './providers/together.js';
import { ElevenLabsClient, createElevenLabsModels } from './providers/elevenlabs.js';
import { createPrunaModels } from './providers/pruna.js';
import { createFalModels } from './providers/fal.js';

const pkg = createRequire(import.meta.url)('../package.json') as {
  name: string;
  version: string;
};

const DEFAULT_IMAGE_EDGE = 750;

const config = loadConfig();
const registry = new ModelRegistry();
const elevenlabs = config.elevenlabsApiKey ? new ElevenLabsClient(config.elevenlabsApiKey) : undefined;

if (config.openaiApiKey) registry.register(...createOpenAIModels(config.openaiApiKey));
if (config.geminiApiKey) registry.register(...createGeminiModels(config.geminiApiKey));
if (config.togetherApiKey) registry.register(...createTogetherModels(config.togetherApiKey));
if (elevenlabs) registry.register(...createElevenLabsModels(elevenlabs));
if (config.prunaApiKey) registry.register(...createPrunaModels(config.prunaApiKey));
if (config.falApiKey) registry.register(...createFalModels(config.falApiKey));

if (registry.list().length === 0) {
  console.error('No API keys found. Copy .env.example to .env and add at least one key.');
  process.exit(1);
}

const server = new McpServer({ name: pkg.name, version: pkg.version });

const textAliases = registry.aliases('text').join(', ');
const imageAliases = registry.aliases('image').join(', ');
const soundAliases = registry.aliases('sound');
const musicAliases = registry.aliases('music');

function errorResult(prefix: string, error: unknown) {
  return {
    content: [
      {
        type: 'text' as const,
        text: `${prefix}: ${error instanceof Error ? error.message : 'Unknown error'}`,
      },
    ],
    isError: true,
  };
}

/** Resolve where to write a generated file, appending the real extension when the caller left it off. */
function outputPath(requested: string | undefined, model: string, extension: string): string {
  if (!requested) return resolve(`polymodel-${model}-${Date.now()}.${extension}`);
  return resolve(/\.[A-Za-z0-9]+$/.test(basename(requested)) ? requested : `${requested}.${extension}`);
}

/** Read images from disk for a model, rejecting formats every provider can't take and counts over its cap. */
function loadImages(paths: string[] | undefined, info: ModelInfo): InputImage[] | undefined {
  if (!paths?.length) return undefined;
  if (!info.maxInputImages) throw new Error(`${info.alias} does not accept images`);
  if (paths.length > info.maxInputImages) {
    throw new Error(`${info.alias} accepts at most ${info.maxInputImages} images; got ${paths.length}`);
  }
  return paths.map((requested) => {
    const path = resolve(requested);
    const bytes = readFileSync(path);
    const sniffed = sniffImage(bytes);
    if (!sniffed || sniffed.extension === 'gif') throw new Error(`${path} is not a PNG, JPEG or WebP image`);
    return { bytes, mimeType: sniffed.mimeType };
  });
}

const imagePathsSchema = z.array(z.string()).optional();

function extractZip(bytes: Buffer, dir: string): string[] {
  mkdirSync(dir, { recursive: true });
  return readZip(bytes).map((entry) => {
    const path = join(dir, basename(entry.name));
    writeFileSync(path, entry.bytes);
    return path;
  });
}

/** Write audio to disk; ZIP archives of several tracks are unpacked into a directory instead. */
function saveAudio(audio: AudioFile, requested: string | undefined, model: string): string[] {
  if (audio.extension === 'zip') {
    const dir = resolve(requested ?? `polymodel-${model}-${Date.now()}`);
    return extractZip(audio.bytes, dir);
  }
  const path = outputPath(requested, model, audio.extension);
  writeFileSync(path, audio.bytes);
  return [path];
}

function audioContent(audio: AudioFile, returnAudio: boolean) {
  return returnAudio && audio.extension !== 'zip'
    ? [{ type: 'audio' as const, data: audio.bytes.toString('base64'), mimeType: audio.mimeType }]
    : [];
}

if (textAliases) {
  server.registerTool(
    'generate_text',
    {
      title: 'Generate text',
      description: `Query an external text model with a prompt and optional images. Available models: ${textAliases}`,
      inputSchema: z.object({
        model: z.string().describe(`Model alias to use (${textAliases})`),
        prompt: z.string().describe('The prompt to send to the model'),
        images: imagePathsSchema.describe(
          "Paths to PNG, JPEG or WebP images to send with the prompt, in order; list_models gives each model's limit"
        ),
        max_tokens: z.number().optional().describe('Maximum tokens in the response'),
        temperature: z.number().min(0).max(2).optional().describe('Sampling temperature'),
        reasoning_effort: z
          .enum(['none', 'low', 'medium', 'high', 'xhigh', 'max'])
          .optional()
          .describe('Reasoning effort, for models that support it'),
      }),
    },
    async ({ model, prompt, images, max_tokens, temperature, reasoning_effort }) => {
      try {
        const textModel = registry.getText(model);
        const result = await textModel.generateText({
          prompt,
          images: loadImages(images, textModel.info),
          maxTokens: max_tokens,
          temperature,
          reasoningEffort: reasoning_effort,
        });
        return { content: [{ type: 'text', text: result.text }] };
      } catch (error) {
        return errorResult(`Error querying ${model}`, error);
      }
    }
  );
}

if (imageAliases) {
  server.registerTool(
    'generate_image',
    {
      title: 'Generate image',
      description:
        `Generate an image with an external model. Available models: ${imageAliases}. ` +
        `Defaults to a ${DEFAULT_IMAGE_EDGE}x${DEFAULT_IMAGE_EDGE} square; providers snap the size to their nearest supported dimensions. ` +
        'Set gif for an animated GIF on models that support it (see list_models). ' +
        'The image is saved to disk and the file path returned.',
      inputSchema: z.object({
        model: z.string().describe(`Model alias to use (${imageAliases})`),
        prompt: z
          .string()
          .describe('Description of the image to generate; refer to reference images by their order ("the first image")'),
        reference_images: imagePathsSchema.describe(
          "Paths to PNG, JPEG or WebP images to edit, combine or take style and subjects from; list_models gives each model's limit"
        ),
        width: z
          .number()
          .int()
          .positive()
          .optional()
          .describe(`Requested width in pixels; defaults to ${DEFAULT_IMAGE_EDGE}, or the model's choice for a GIF`),
        height: z
          .number()
          .int()
          .positive()
          .optional()
          .describe(`Requested height in pixels; defaults to ${DEFAULT_IMAGE_EDGE}, or the model's choice for a GIF`),
        gif: z
          .boolean()
          .default(false)
          .describe('Generate an animated GIF; leave width and height unset unless you need a size, as large frames can exceed the provider limit'),
        frame_count: z.number().int().min(2).max(16).optional().describe('GIF frames; the provider defaults to 9'),
        frame_duration_ms: z
          .number()
          .int()
          .min(20)
          .max(2000)
          .optional()
          .describe('Delay per GIF frame in milliseconds; the provider defaults to 200'),
        output_path: z
          .string()
          .optional()
          .describe('Where to save the image; defaults to polymodel-<model>-<n>.<format> in the current directory'),
        return_image: z
          .boolean()
          .default(false)
          .describe('Also return the image content inline (costs context; the file is written regardless)'),
      }),
    },
    async ({ model, prompt, reference_images, width, height, gif, frame_count, frame_duration_ms, output_path, return_image }) => {
      try {
        const imageModel = registry.getImage(model);
        if (gif && !imageModel.generateGif) throw new Error(`${imageModel.info.alias} does not support GIF output`);
        if (!gif && (frame_count !== undefined || frame_duration_ms !== undefined)) {
          throw new Error('frame_count and frame_duration_ms require gif: true');
        }
        const referenceImages = loadImages(reference_images, imageModel.info);
        const hasSize = width !== undefined || height !== undefined;
        const sized = { width: width ?? DEFAULT_IMAGE_EDGE, height: height ?? DEFAULT_IMAGE_EDGE };
        const result = gif
          ? await imageModel.generateGif!({
              prompt,
              referenceImages,
              ...(hasSize ? sized : {}),
              frameCount: frame_count,
              frameDurationMs: frame_duration_ms,
            })
          : await imageModel.generateImage({ prompt, referenceImages, ...sized });
        const bytes = Buffer.from(result.base64, 'base64');
        const sniffed = sniffImage(bytes);
        const mimeType = sniffed?.mimeType ?? result.mimeType;
        const path = outputPath(output_path, model, sniffed?.extension ?? 'png');
        writeFileSync(path, bytes);

        const actualSize =
          sniffed?.width && sniffed?.height
            ? `${sniffed.width}x${sniffed.height}`
            : `${result.width}x${result.height}`;
        const summary = [
          `Saved image to ${path}`,
          `model: ${result.model}`,
          `format: ${mimeType}`,
          `size: ${actualSize}`,
          ...(result.sizeNote ? [`note: ${result.sizeNote}`] : []),
        ].join('\n');

        return {
          content: [
            { type: 'text' as const, text: summary },
            ...(return_image ? [{ type: 'image' as const, data: result.base64, mimeType }] : []),
          ],
        };
      } catch (error) {
        return errorResult(`Error generating image with ${model}`, error);
      }
    }
  );
}

const AUDIO_FORMATS = [
  'mp3_22050_32',
  'mp3_24000_48',
  'mp3_44100_32',
  'mp3_44100_64',
  'mp3_44100_96',
  'mp3_44100_128',
  'mp3_44100_192',
  'pcm_8000',
  'pcm_16000',
  'pcm_22050',
  'pcm_24000',
  'pcm_32000',
  'pcm_44100',
  'pcm_48000',
  'ulaw_8000',
  'alaw_8000',
  'opus_48000_32',
  'opus_48000_64',
  'opus_48000_96',
  'opus_48000_128',
  'opus_48000_192',
] as const;
const MUSIC_FORMATS = ['auto', 'mp3_48000_128', 'mp3_48000_192', 'mp3_48000_240', 'mp3_48000_320', ...AUDIO_FORMATS] as const;
const FORMAT_NOTE =
  'codec_samplerate_bitrate. PCM is raw 16-bit little-endian; u-law and a-law are raw too. ' +
  'Higher MP3 bitrates and 44.1 kHz PCM need a higher ElevenLabs subscription tier';

const outputPathSchema = z
  .string()
  .optional()
  .describe('Where to save the audio; defaults to polymodel-<model>-<n>.<ext> in the current directory');
const returnAudioSchema = z
  .boolean()
  .default(false)
  .describe('Also return the audio inline (costs context; the file is written regardless)');
const c2paSchema = z.boolean().optional().describe('Sign the output with C2PA content credentials (MP3 only)');
const musicModelSchema = z
  .string()
  .default(musicAliases[0] ?? '')
  .describe(`Model alias to use (${musicAliases.join(', ')}); the plan format depends on the model`);

if (soundAliases.length) {
  server.registerTool(
    'generate_sound_effect',
    {
      title: 'Generate sound effect',
      description:
        `Generate a sound effect from a text description. Available models: ${soundAliases.join(', ')}. ` +
        'The audio is saved to disk and the file path returned.',
      inputSchema: z.object({
        model: z.string().default(soundAliases[0]).describe(`Model alias to use (${soundAliases.join(', ')})`),
        prompt: z.string().describe('Description of the sound, e.g. "heavy wooden door creaking open in a stone hall"'),
        duration_seconds: z
          .number()
          .min(0.5)
          .max(30)
          .optional()
          .describe('Length in seconds (0.5-30); the model picks one from the prompt when unset'),
        loop: z.boolean().optional().describe('Make the sound loop seamlessly'),
        prompt_influence: z
          .number()
          .min(0)
          .max(1)
          .optional()
          .describe('0-1 (default 0.3); higher follows the prompt more closely with less variation'),
        output_format: z.enum(AUDIO_FORMATS).optional().describe(`Default mp3_44100_128. ${FORMAT_NOTE}`),
        output_path: outputPathSchema,
        return_audio: returnAudioSchema,
      }),
    },
    async ({ model, prompt, duration_seconds, loop, prompt_influence, output_format, output_path, return_audio }) => {
      try {
        const result = await registry.getSound(model).generateSound({
          prompt,
          durationSeconds: duration_seconds,
          loop,
          promptInfluence: prompt_influence,
          outputFormat: output_format,
        });
        const [path] = saveAudio(result.audio, output_path, model);
        const summary = [
          `Saved sound effect to ${path}`,
          `model: ${result.model}`,
          `format: ${output_format ?? result.audio.mimeType}`,
          ...(result.cost ? [`cost: ${result.cost} characters`] : []),
        ].join('\n');
        return { content: [{ type: 'text' as const, text: summary }, ...audioContent(result.audio, return_audio)] };
      } catch (error) {
        return errorResult(`Error generating sound effect with ${model}`, error);
      }
    }
  );
}

if (musicAliases.length) {
  const timeRange = z.object({ start_ms: z.number().int().min(0), end_ms: z.number().int().min(0) });
  const audioRef = z.looseObject({
    song_id: z.string().describe('Id of a stored song (from generate_music with store_for_inpainting, or upload_music)'),
    range: timeRange,
  });
  const generationChunk = z.looseObject({
    text: z
      .string()
      .describe(
        'Optional section name in brackets on the first line ("[Verse 1]"), then lyric lines and inline directions in braces ("{whispered}"). Max 30 lines of 200 characters'
      ),
    duration_ms: z.number().int().min(3000).max(120000),
    positive_styles: z
      .array(z.string())
      .max(50)
      .describe('Styles to include. The first chunk sets the genre and tone; give it 6-7 or more'),
    negative_styles: z.array(z.string()).max(50).optional().describe('Styles to avoid'),
    context_adherence: z
      .enum(['low', 'medium', 'high'])
      .optional()
      .describe('How closely this chunk follows its neighbors (default high)'),
    conditioning_ref: audioRef
      .optional()
      .describe('Slice of a stored song (max 30 s) to condition this chunk on; set it on the first chunk to steer the whole song'),
    condition_strength: z.enum(['low', 'medium', 'high', 'xhigh']).optional(),
  });
  const chunkPlan = z
    .looseObject({
      chunks: z
        .array(z.union([generationChunk, audioRef.describe('Insert this slice of a stored song unchanged')]))
        .min(1)
        .max(30),
    })
    .describe('Plan for eleven-music and eleven-music-v2: up to 30 chunks, each generated or copied from a stored song');
  const sectionPlan = z
    .looseObject({
      positive_global_styles: z.array(z.string()),
      negative_global_styles: z.array(z.string()),
      sections: z.array(
        z.looseObject({
          section_name: z.string().min(1).max(100),
          positive_local_styles: z.array(z.string()),
          negative_local_styles: z.array(z.string()),
          duration_ms: z.number().int().min(3000).max(120000),
          lines: z.array(z.string().max(200)).max(30).describe('Lyrics; empty for instrumental sections'),
          source_from: z
            .looseObject({ song_id: z.string(), range: timeRange, negative_ranges: z.array(timeRange).max(10).optional() })
            .optional()
            .describe('Take this section from a stored song (inpainting)'),
        })
      ),
    })
    .describe('Plan for eleven-music-v1');
  const compositionPlanSchema = z.union([chunkPlan, sectionPlan]);

  server.registerTool(
    'generate_music',
    {
      title: 'Generate music',
      description:
        `Compose a song or instrumental from a prompt or a composition plan. Available models: ${musicAliases.join(', ')}. ` +
        'Give either prompt or composition_plan, not both. To edit a song, store it for inpainting, then send a plan that copies ' +
        'the parts to keep (audio reference chunks) and regenerates the rest. The audio is saved to disk, with the plan the model ' +
        'used and song metadata in a .json file beside it.',
      inputSchema: z.object({
        model: musicModelSchema,
        prompt: z.string().max(4100).optional().describe('Free-text description of the music, optionally with lyrics'),
        composition_plan: compositionPlanSchema
          .optional()
          .describe('Structured plan; create_music_plan drafts one from a prompt'),
        music_length_ms: z
          .number()
          .int()
          .min(3000)
          .max(600000)
          .optional()
          .describe('Length with a prompt (3 s-10 min); the model picks one when unset. Plans set length per chunk'),
        generation_mode: z
          .enum(['track', 'loop', 'ambience', 'video_to_music'])
          .optional()
          .describe('Hint for prompt-based generation, e.g. loop for a seamless loop or ambience for background beds'),
        force_instrumental: z.boolean().optional().describe('Guarantee no vocals (prompt only)'),
        seed: z
          .number()
          .int()
          .min(0)
          .max(2147483647)
          .optional()
          .describe('Seed for more repeatable results (composition_plan only)'),
        finetune_id: z.string().optional().describe('Finetune to generate with; see manage_music_finetunes'),
        finetune_strength: z
          .number()
          .gt(0)
          .max(2)
          .optional()
          .describe('How strongly the finetune applies (default 1); lower leaves more room for the prompt'),
        use_phonetic_names: z
          .boolean()
          .optional()
          .describe('Spell proper names phonetically in lyrics for better pronunciation'),
        respect_sections_durations: z
          .boolean()
          .optional()
          .describe('eleven-music-v1 plans only: false lets the model adjust section lengths while keeping the total'),
        store_for_inpainting: z
          .boolean()
          .optional()
          .describe('Store the song so later plans can reuse or condition on parts of it by song_id'),
        with_timestamps: z.boolean().optional().describe('Include word-level lyric timestamps in the .json file'),
        with_waveform_visual: z
          .boolean()
          .optional()
          .describe('Include a low-resolution waveform (4 values per second) in the .json file'),
        sign_with_c2pa: c2paSchema,
        output_format: z
          .enum(MUSIC_FORMATS)
          .optional()
          .describe(`Default auto (mp3_48000_192 for v2 models, mp3_44100_128 for v1). ${FORMAT_NOTE}`),
        output_path: outputPathSchema,
        return_audio: returnAudioSchema,
      }),
    },
    async (args) => {
      const { model, output_path, return_audio } = args;
      try {
        if (!args.prompt === !args.composition_plan) {
          throw new Error('Give exactly one of prompt or composition_plan');
        }
        const result = await registry.getMusic(model).compose({
          prompt: args.prompt,
          compositionPlan: args.composition_plan,
          lengthMs: args.music_length_ms,
          generationMode: args.generation_mode,
          forceInstrumental: args.force_instrumental,
          seed: args.seed,
          finetuneId: args.finetune_id,
          finetuneStrength: args.finetune_strength,
          usePhoneticNames: args.use_phonetic_names,
          respectSectionsDurations: args.respect_sections_durations,
          storeForInpainting: args.store_for_inpainting,
          withTimestamps: args.with_timestamps,
          withWaveformVisual: args.with_waveform_visual,
          signWithC2pa: args.sign_with_c2pa,
          outputFormat: args.output_format,
        });
        const [path] = saveAudio(result.audio, output_path, model);
        const lines = [`Saved music to ${path}`, `model: ${result.model}`, `format: ${result.audio.mimeType}`];
        if (result.songId) {
          lines.push(`song_id: ${result.songId}${args.store_for_inpainting ? ' (stored for inpainting)' : ''}`);
        }
        if (result.details) {
          const detailsPath = path.replace(/\.[^.\\/]+$/, '') + '.json';
          writeFileSync(detailsPath, JSON.stringify({ song_id: result.songId, ...result.details }, null, 2) + '\n');
          const meta = result.details.song_metadata as { title?: string; genres?: string[] } | undefined;
          if (meta?.title) lines.push(`title: ${meta.title}`);
          if (meta?.genres?.length) lines.push(`genres: ${meta.genres.join(', ')}`);
          lines.push(`details: ${detailsPath}`);
          if (result.details.composition_plan) {
            lines.push(`composition_plan: ${JSON.stringify(result.details.composition_plan)}`);
          }
        }
        return {
          content: [{ type: 'text' as const, text: lines.join('\n') }, ...audioContent(result.audio, return_audio)],
        };
      } catch (error) {
        return errorResult(`Error generating music with ${model}`, error);
      }
    }
  );

  server.registerTool(
    'create_music_plan',
    {
      title: 'Create music plan',
      description:
        'Draft a composition plan from a prompt without generating audio, to review or edit before passing it to generate_music. ' +
        'Can also revise an existing plan.',
      inputSchema: z.object({
        model: musicModelSchema,
        prompt: z.string().max(4100).describe('Description of the music, or the changes to make to source_composition_plan'),
        music_length_ms: z.number().int().min(3000).max(600000).optional().describe('Total length (3 s-10 min)'),
        source_composition_plan: compositionPlanSchema.optional().describe('Existing plan to base the new one on'),
      }),
    },
    async ({ model, prompt, music_length_ms, source_composition_plan }) => {
      try {
        const plan = await registry.getMusic(model).createPlan({
          prompt,
          lengthMs: music_length_ms,
          sourcePlan: source_composition_plan,
        });
        return { content: [{ type: 'text' as const, text: JSON.stringify(plan, null, 2) }] };
      } catch (error) {
        return errorResult(`Error creating music plan with ${model}`, error);
      }
    }
  );

  server.registerTool(
    'generate_music_from_video',
    {
      title: 'Generate music from video',
      description:
        'Score one or more video clips: the clips are joined in order and music is generated to fit them. ' +
        'Up to 10 clips, 200 MB and 600 seconds in total. The audio is saved to disk; if several tracks come back they are ' +
        'unpacked into a directory.',
      inputSchema: z.object({
        model: musicModelSchema,
        video_paths: z.array(z.string()).min(1).max(10).describe('Local video files, in playback order'),
        description: z.string().min(1).max(1000).optional().describe('What the music should be like'),
        tags: z.array(z.string()).max(10).optional().describe('Style tags, e.g. ["upbeat", "cinematic"]'),
        sign_with_c2pa: c2paSchema,
        output_format: z.enum(AUDIO_FORMATS).optional().describe(`Default mp3_44100_128. ${FORMAT_NOTE}`),
        output_path: outputPathSchema,
        return_audio: returnAudioSchema,
      }),
    },
    async ({ model, video_paths, description, tags, sign_with_c2pa, output_format, output_path, return_audio }) => {
      try {
        const musicModel = registry.getMusic(model);
        const audio = await musicModel.videoToMusic({
          videoPaths: video_paths.map((p) => resolve(p)),
          description,
          tags,
          signWithC2pa: sign_with_c2pa,
          outputFormat: output_format,
        });
        const paths = saveAudio(audio, output_path, model);
        const text = `Saved music to ${paths.join(', ')}\nmodel: ${musicModel.info.underlyingModel}`;
        return { content: [{ type: 'text' as const, text }, ...audioContent(audio, return_audio)] };
      } catch (error) {
        return errorResult(`Error generating music from video with ${model}`, error);
      }
    }
  );
}

if (elevenlabs && musicAliases.length) {
  server.registerTool(
    'upload_music',
    {
      title: 'Upload music',
      description:
        'Upload an audio file to ElevenLabs and get a song_id, so composition plans can reuse or condition on parts of it ' +
        '(inpainting). Optionally extract its composition plan, lyric timestamps and waveform.',
      inputSchema: z.object({
        file_path: z.string().describe('Local audio file'),
        extract_composition_plan: z
          .boolean()
          .default(false)
          .describe('Return a composition plan for the song (adds latency)'),
        model: musicModelSchema.describe('Plan format to extract, by model alias'),
        with_timestamps: z.boolean().optional().describe('Transcribe the lyrics with word-level timestamps'),
        with_waveform_visual: z.boolean().optional().describe('Return a low-resolution waveform (4 values per second)'),
      }),
    },
    async ({ file_path, extract_composition_plan, model, with_timestamps, with_waveform_visual }) => {
      try {
        const result = await elevenlabs.uploadMusic(resolve(file_path), {
          extractCompositionPlan: extract_composition_plan ? registry.getMusic(model).info.underlyingModel : undefined,
          withTimestamps: with_timestamps,
          withWaveformVisual: with_waveform_visual,
        });
        return { content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }] };
      } catch (error) {
        return errorResult('Error uploading music', error);
      }
    }
  );

  server.registerTool(
    'separate_stems',
    {
      title: 'Separate stems',
      description:
        'Split an audio file into stems and save each one to a directory. six_stems_v1 gives vocals, drums, bass, guitar, ' +
        'piano and other; two_stems_v1 gives vocals and accompaniment. Long files can take several minutes.',
      inputSchema: z.object({
        file_path: z.string().describe('Local audio file'),
        stem_variation: z.enum(['six_stems_v1', 'two_stems_v1']).default('six_stems_v1'),
        output_format: z.enum(AUDIO_FORMATS).optional().describe(`Default mp3_44100_128. ${FORMAT_NOTE}`),
        sign_with_c2pa: c2paSchema,
        output_dir: z
          .string()
          .optional()
          .describe('Directory for the stems; defaults to polymodel-stems-<n> in the current directory'),
      }),
    },
    async ({ file_path, stem_variation, output_format, sign_with_c2pa, output_dir }) => {
      try {
        const zip = await elevenlabs.separateStems(resolve(file_path), {
          stemVariationId: stem_variation,
          outputFormat: output_format,
          signWithC2pa: sign_with_c2pa,
        });
        const paths = extractZip(zip, resolve(output_dir ?? `polymodel-stems-${Date.now()}`));
        return { content: [{ type: 'text' as const, text: `Saved ${paths.length} stems:\n${paths.join('\n')}` }] };
      } catch (error) {
        return errorResult('Error separating stems', error);
      }
    }
  );

  server.registerTool(
    'manage_music_finetunes',
    {
      title: 'Manage music finetunes',
      description:
        'List, inspect, create, update or delete ElevenLabs music finetunes (custom models trained on your audio). ' +
        'Pass a completed finetune id to generate_music as finetune_id. Training runs in the background; use get to check status.',
      annotations: { destructiveHint: true },
      inputSchema: z.object({
        action: z.enum(['list', 'get', 'create', 'update', 'delete']),
        finetune_id: z.string().optional().describe('Required for get, update and delete'),
        name: z.string().min(5).max(200).optional().describe('create (required) and update'),
        primary_genre: z.string().optional().describe('create (required) and update'),
        tags: z.array(z.string()).optional().describe('create and update (update allows at most 5)'),
        visibility: z
          .enum(['private', 'workspace', 'public'])
          .optional()
          .describe('create/update: private or workspace. list: filter, where public means ElevenLabs-curated'),
        file_paths: z.array(z.string()).max(50).optional().describe('create: local audio files to train on (max 50)'),
        model: musicModelSchema.describe('create: base model to finetune, by model alias'),
        created_by: z.enum(['self', 'workspace', 'elevenlabs']).optional().describe('list: filter by creator'),
        sort: z.enum(['created_at', 'name']).optional().describe('list: sort field (default created_at)'),
        sort_direction: z.enum(['asc', 'desc']).optional().describe('list: default desc'),
        page_size: z.number().int().min(1).max(150).optional().describe('list: default 50'),
        cursor: z.string().optional().describe('list: next_cursor from the previous page'),
      }),
    },
    async (args) => {
      try {
        const id = args.finetune_id;
        const needId = () => {
          if (!id) throw new Error(`finetune_id is required for ${args.action}`);
          return id;
        };
        const fields = {
          name: args.name,
          primaryGenre: args.primary_genre,
          tags: args.tags,
          visibility: args.visibility,
        };
        let result: unknown;
        switch (args.action) {
          case 'list':
            result = await elevenlabs.listFinetunes({
              cursor: args.cursor,
              pageSize: args.page_size,
              visibility: args.visibility,
              createdBy: args.created_by,
              sort: args.sort,
              sortDirection: args.sort_direction,
            });
            break;
          case 'get':
            result = await elevenlabs.getFinetune(needId());
            break;
          case 'create':
            if (!args.name || !args.primary_genre) throw new Error('name and primary_genre are required for create');
            result = await elevenlabs.createFinetune(
              fields,
              registry.getMusic(args.model).info.underlyingModel,
              (args.file_paths ?? []).map((p) => resolve(p))
            );
            break;
          case 'update':
            result = await elevenlabs.updateFinetune(needId(), fields);
            break;
          case 'delete':
            result = await elevenlabs.deleteFinetune(needId());
            break;
        }
        return { content: [{ type: 'text' as const, text: JSON.stringify(result, null, 2) }] };
      } catch (error) {
        return errorResult(`Error running finetune ${args.action}`, error);
      }
    }
  );
}

server.registerTool(
  'list_models',
  {
    title: 'List models',
    description:
      'List all available models with their provider, modality, underlying model id, and how many input images each accepts',
    inputSchema: z.object({}),
    outputSchema: z.object({
      models: z.array(
        z.object({
          alias: z.string(),
          provider: z.string(),
          modality: z.enum(['text', 'image', 'sound', 'music']),
          underlyingModel: z.string(),
          description: z.string(),
          maxInputImages: z.number().optional(),
        })
      ),
    }),
  },
  async () => {
    const output = { models: registry.list() };
    return {
      content: [{ type: 'text', text: JSON.stringify(output, null, 2) }],
      structuredContent: output,
    };
  }
);

server.registerTool(
  'get_version',
  {
    title: 'Get version',
    description: 'Get the name and version of this MCP server',
    inputSchema: z.object({}),
  },
  async () => ({
    content: [{ type: 'text', text: JSON.stringify({ name: pkg.name, version: pkg.version }, null, 2) }],
  })
);

async function main() {
  // Default is 10 MiB; raised so large prompts never trip the transport.
  await server.connect(new StdioServerTransport(undefined, undefined, { maxBufferSize: 64 * 1024 * 1024 }));
  console.error('Polymodel MCP Server running on stdio');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
