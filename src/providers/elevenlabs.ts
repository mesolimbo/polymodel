import { openAsBlob } from 'fs';
import { basename, extname } from 'path';
import { Agent, FormData, fetch } from 'undici';
import {
  AudioFile,
  CompositionPlan,
  ModelInfo,
  MusicModel,
  MusicPlanRequest,
  MusicRequest,
  MusicResult,
  SoundModel,
  SoundRequest,
  SoundResult,
  VideoToMusicRequest,
} from '../domain/types.js';

const BASE_URL = 'https://api.elevenlabs.io/v1';
const SOUND_MODEL = 'eleven_text_to_sound_v3';

// Long songs, stem separation and video scoring can run past undici's default
// five-minute header and body timeouts, so ElevenLabs calls wait indefinitely.
const dispatcher = new Agent({ headersTimeout: 0, bodyTimeout: 0 });

const CODECS: Record<string, { extension: string; mimeType: string }> = {
  mp3: { extension: 'mp3', mimeType: 'audio/mpeg' },
  pcm: { extension: 'pcm', mimeType: 'audio/pcm' },
  ulaw: { extension: 'ulaw', mimeType: 'audio/basic' },
  alaw: { extension: 'alaw', mimeType: 'audio/x-alaw-basic' },
  opus: { extension: 'opus', mimeType: 'audio/opus' },
  zip: { extension: 'zip', mimeType: 'application/zip' },
};

const UPLOAD_TYPES: Record<string, string> = {
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
  '.opus': 'audio/opus',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
};

/** Raw PCM, u-law and a-law come back headerless, so the extension follows the requested codec. */
function audioFile(bytes: Buffer, outputFormat?: string, contentType?: string | null, filename?: string): AudioFile {
  if (contentType?.includes('zip')) return { bytes, ...CODECS.zip };
  const codec =
    CODECS[filename ? extname(filename).slice(1).toLowerCase() : ''] ??
    CODECS[outputFormat?.split('_')[0] ?? ''] ??
    CODECS.mp3;
  return { bytes, ...codec };
}

interface MultipartPart {
  headers: Record<string, string>;
  body: Buffer;
}

function parseMultipart(body: Buffer, contentType: string): MultipartPart[] {
  const boundary = /boundary="?([^";]+)"?/i.exec(contentType)?.[1];
  if (!boundary) throw new Error(`ElevenLabs sent multipart data without a boundary (${contentType})`);
  const delimiter = Buffer.from(`--${boundary}`);
  const parts: MultipartPart[] = [];
  let start = body.indexOf(delimiter);
  while (start >= 0) {
    const after = start + delimiter.length;
    if (body.toString('latin1', after, after + 2) === '--') break;
    const next = body.indexOf(delimiter, after);
    if (next < 0) break;
    // Each part sits between the CRLF ending the delimiter line and the CRLF before the next one.
    const part = body.subarray(after + 2, next - 2);
    const headerEnd = part.indexOf('\r\n\r\n');
    const headers: Record<string, string> = {};
    for (const line of part.toString('latin1', 0, headerEnd).split('\r\n')) {
      const colon = line.indexOf(':');
      if (colon > 0) headers[line.slice(0, colon).trim().toLowerCase()] = line.slice(colon + 1).trim();
    }
    parts.push({ headers, body: part.subarray(headerEnd + 4) });
    start = next;
  }
  return parts;
}

async function appendFile(form: FormData, field: string, path: string): Promise<void> {
  const blob = await openAsBlob(path, { type: UPLOAD_TYPES[extname(path).toLowerCase()] ?? '' });
  form.append(field, blob, basename(path));
}

type Query = Record<string, string | number | undefined>;
type Fields = Record<string, string | number | boolean | string[] | undefined>;

function appendFields(form: FormData, fields: Fields): void {
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined) continue;
    for (const item of Array.isArray(value) ? value : [value]) form.append(key, String(item));
  }
}

export interface UploadMusicOptions {
  extractCompositionPlan?: string;
  withTimestamps?: boolean;
  withWaveformVisual?: boolean;
}

export interface StemOptions {
  stemVariationId?: string;
  outputFormat?: string;
  signWithC2pa?: boolean;
}

export interface FinetuneListOptions {
  cursor?: string;
  pageSize?: number;
  visibility?: string;
  createdBy?: string;
  sort?: string;
  sortDirection?: string;
}

export interface FinetuneFields {
  name?: string;
  primaryGenre?: string;
  tags?: string[];
  visibility?: string;
}

/** Thin HTTP client for the ElevenLabs endpoints that are not tied to one generation model. */
export class ElevenLabsClient {
  constructor(private apiKey: string) {}

  async request(
    method: string,
    path: string,
    { query, json, form }: { query?: Query; json?: unknown; form?: FormData } = {}
  ) {
    const url = new URL(`${BASE_URL}${path}`);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
    const response = await fetch(url, {
      method,
      dispatcher,
      headers: {
        'xi-api-key': this.apiKey,
        ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}),
      },
      body: json !== undefined ? JSON.stringify(json) : form,
    });
    if (!response.ok) {
      throw new Error(`ElevenLabs API error ${response.status}: ${await response.text()}`);
    }
    return response;
  }

  async uploadMusic(path: string, opts: UploadMusicOptions): Promise<Record<string, unknown>> {
    const form = new FormData();
    await appendFile(form, 'file', path);
    appendFields(form, {
      extract_composition_plan: opts.extractCompositionPlan,
      with_timestamps: opts.withTimestamps,
      with_waveform_visual: opts.withWaveformVisual,
    });
    const response = await this.request('POST', '/music/upload', { form });
    return (await response.json()) as Record<string, unknown>;
  }

  /** Returns a ZIP archive with one audio file per stem. */
  async separateStems(path: string, opts: StemOptions): Promise<Buffer> {
    const form = new FormData();
    await appendFile(form, 'file', path);
    appendFields(form, { stem_variation_id: opts.stemVariationId, sign_with_c2pa: opts.signWithC2pa });
    const response = await this.request('POST', '/music/stem-separation', {
      query: { output_format: opts.outputFormat },
      form,
    });
    return Buffer.from(await response.arrayBuffer());
  }

  async listFinetunes(opts: FinetuneListOptions): Promise<unknown> {
    const response = await this.request('GET', '/music/finetunes', {
      query: {
        cursor: opts.cursor,
        page_size: opts.pageSize,
        visibility: opts.visibility,
        created_by: opts.createdBy,
        sort: opts.sort,
        sort_direction: opts.sortDirection,
      },
    });
    return response.json();
  }

  async getFinetune(id: string): Promise<unknown> {
    return (await this.request('GET', `/music/finetunes/${encodeURIComponent(id)}`)).json();
  }

  async createFinetune(fields: FinetuneFields, modelId: string, filePaths: string[]): Promise<unknown> {
    const form = new FormData();
    appendFields(form, {
      name: fields.name,
      primary_genre: fields.primaryGenre,
      tags: fields.tags,
      visibility: fields.visibility,
      model_id: modelId,
    });
    for (const path of filePaths) await appendFile(form, 'files', path);
    return (await this.request('POST', '/music/finetunes', { form })).json();
  }

  async updateFinetune(id: string, fields: FinetuneFields): Promise<unknown> {
    const json = {
      name: fields.name,
      primary_genre: fields.primaryGenre,
      tags: fields.tags,
      visibility: fields.visibility,
    };
    return (await this.request('PATCH', `/music/finetunes/${encodeURIComponent(id)}`, { json })).json();
  }

  async deleteFinetune(id: string): Promise<unknown> {
    return (await this.request('DELETE', `/music/finetunes/${encodeURIComponent(id)}`)).json();
  }
}

export class ElevenSoundModel implements SoundModel {
  readonly info: ModelInfo = {
    alias: 'eleven-sfx',
    provider: 'elevenlabs',
    modality: 'sound',
    underlyingModel: SOUND_MODEL,
    description: 'ElevenLabs sound effects; 0.5-30 seconds (auto when unset), optional seamless loop',
  };

  constructor(private client: ElevenLabsClient) {}

  async generateSound(req: SoundRequest): Promise<SoundResult> {
    const response = await this.client.request('POST', '/sound-generation', {
      query: { output_format: req.outputFormat },
      json: {
        text: req.prompt,
        model_id: SOUND_MODEL,
        loop: req.loop,
        duration_seconds: req.durationSeconds,
        prompt_influence: req.promptInfluence,
      },
    });
    const bytes = Buffer.from(await response.arrayBuffer());
    return {
      audio: audioFile(bytes, req.outputFormat, response.headers.get('content-type')),
      model: SOUND_MODEL,
      cost: response.headers.get('character-cost') ?? undefined,
    };
  }
}

export class ElevenMusicModel implements MusicModel {
  constructor(
    private client: ElevenLabsClient,
    readonly info: ModelInfo
  ) {}

  private get modelId(): string {
    return this.info.underlyingModel;
  }

  async compose(req: MusicRequest): Promise<MusicResult> {
    // The detailed endpoint costs the same and also returns the plan it used,
    // which callers need to edit the song afterwards.
    const response = await this.client.request('POST', '/music/detailed', {
      query: { output_format: req.outputFormat },
      json: {
        model_id: this.modelId,
        prompt: req.prompt,
        composition_plan: req.compositionPlan,
        music_length_ms: req.lengthMs,
        generation_mode: req.generationMode,
        force_instrumental: req.forceInstrumental,
        seed: req.seed,
        finetune_id: req.finetuneId,
        finetune_strength: req.finetuneStrength,
        use_phonetic_names: req.usePhoneticNames,
        respect_sections_durations: req.respectSectionsDurations,
        store_for_inpainting: req.storeForInpainting,
        with_timestamps: req.withTimestamps,
        with_waveform_visual: req.withWaveformVisual,
        sign_with_c2pa: req.signWithC2pa,
      },
    });
    const contentType = response.headers.get('content-type') ?? '';
    const body = Buffer.from(await response.arrayBuffer());
    const songId = response.headers.get('song-id') ?? undefined;
    if (!contentType.startsWith('multipart/')) {
      return { audio: audioFile(body, req.outputFormat, contentType), model: this.modelId, songId };
    }

    const parts = parseMultipart(body, contentType);
    const jsonPart = parts.find((p) => p.headers['content-type']?.includes('json'));
    const audioPart = parts.find((p) => p !== jsonPart);
    if (!audioPart) throw new Error('ElevenLabs returned no audio in the detailed response');
    const filename = /filename="?([^";]+)"?/.exec(audioPart.headers['content-disposition'] ?? '')?.[1];
    return {
      audio: audioFile(audioPart.body, req.outputFormat, audioPart.headers['content-type'], filename),
      model: this.modelId,
      songId,
      details: jsonPart ? (JSON.parse(jsonPart.body.toString('utf8')) as Record<string, unknown>) : undefined,
    };
  }

  async createPlan(req: MusicPlanRequest): Promise<CompositionPlan> {
    const response = await this.client.request('POST', '/music/plan', {
      json: {
        model_id: this.modelId,
        prompt: req.prompt,
        music_length_ms: req.lengthMs,
        source_composition_plan: req.sourcePlan,
      },
    });
    return (await response.json()) as CompositionPlan;
  }

  async videoToMusic(req: VideoToMusicRequest): Promise<AudioFile> {
    const form = new FormData();
    for (const path of req.videoPaths) await appendFile(form, 'videos', path);
    appendFields(form, {
      description: req.description,
      tags: req.tags,
      model_id: this.modelId,
      sign_with_c2pa: req.signWithC2pa,
    });
    const response = await this.client.request('POST', '/music/video-to-music', {
      query: { output_format: req.outputFormat },
      form,
    });
    const bytes = Buffer.from(await response.arrayBuffer());
    return audioFile(bytes, req.outputFormat, response.headers.get('content-type'));
  }
}

export function createElevenLabsModels(client: ElevenLabsClient): [SoundModel, ...MusicModel[]] {
  const music = (alias: string, underlyingModel: string, description: string) =>
    new ElevenMusicModel(client, { alias, provider: 'elevenlabs', modality: 'music', underlyingModel, description });
  return [
    new ElevenSoundModel(client),
    music(
      'eleven-music',
      'music_v2_5',
      'ElevenLabs Music v2.5; 3 s-10 min from a prompt or a chunk-based composition plan, with inpainting and audio conditioning'
    ),
    music('eleven-music-v2', 'music_v2', 'ElevenLabs Music v2; same chunk-based plan format as v2.5'),
    music(
      'eleven-music-v1',
      'music_v1',
      'ElevenLabs Music v1 (deprecated by ElevenLabs); section-based plan format with global and local styles'
    ),
  ];
}
