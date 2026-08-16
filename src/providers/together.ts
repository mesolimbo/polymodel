import OpenAI from 'openai';
import {
  ImageModel,
  ImageRequest,
  ImageResult,
  ModelInfo,
  TextModel,
  TextRequest,
  TextResult,
} from '../domain/types.js';

const BASE_URL = 'https://api.together.xyz/v1';
const TEXT_MODEL = 'moonshotai/Kimi-K3';
const IMAGE_MODEL = 'black-forest-labs/FLUX.2-dev';

// FLUX.2 dev accepts direct pixel dimensions; snap to multiples of 16 within
// the ranges Together serves.
const EDGE_STEP = 16;
const MIN_EDGE = 512;
const MAX_EDGE = 2048;

function normalizeSize(width: number, height: number): { width: number; height: number } {
  const snap = (v: number) =>
    Math.min(MAX_EDGE, Math.max(MIN_EDGE, Math.round(v / EDGE_STEP) * EDGE_STEP));
  return { width: snap(width), height: snap(height) };
}

export class KimiTextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'kimi-k3',
    provider: 'together',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `Moonshot Kimi K3 served by Together AI`,
  };

  constructor(private client: OpenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const response = await this.client.chat.completions.create({
      model: TEXT_MODEL,
      messages: [{ role: 'user', content: req.prompt }],
      max_completion_tokens: req.maxTokens ?? 16384,
      ...(req.reasoningEffort
        ? { reasoning_effort: req.reasoningEffort as 'low' | 'high' }
        : {}),
    });
    return {
      text: response.choices[0]?.message?.content || 'No response received',
      model: TEXT_MODEL,
    };
  }
}

export class FluxImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'flux-2-dev',
    provider: 'together',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `FLUX.2 [dev] served by Together AI; edges snap to multiples of 16 between 512 and 2048`,
  };

  constructor(private apiKey: string) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const { width, height } = normalizeSize(req.width, req.height);
    const response = await fetch(`${BASE_URL}/images/generations`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: IMAGE_MODEL,
        prompt: req.prompt,
        width,
        height,
        n: 1,
        response_format: 'base64',
      }),
    });
    if (!response.ok) {
      throw new Error(`Together API error ${response.status}: ${await response.text()}`);
    }

    const payload = (await response.json()) as {
      data?: Array<{ b64_json?: string; url?: string }>;
    };
    const item = payload.data?.[0];
    let base64 = item?.b64_json;
    if (!base64 && item?.url) {
      const download = await fetch(item.url);
      if (!download.ok) throw new Error(`Failed to download image: ${download.status}`);
      base64 = Buffer.from(await download.arrayBuffer()).toString('base64');
    }
    if (!base64) throw new Error('Together returned no image data');

    return {
      base64,
      mimeType: 'image/png',
      width,
      height,
      model: IMAGE_MODEL,
      sizeNote:
        width !== req.width || height !== req.height
          ? `requested ${req.width}x${req.height}, adjusted to ${width}x${height} (FLUX.2 dev size constraints)`
          : undefined,
    };
  }
}

export function createTogetherModels(apiKey: string): [TextModel, ImageModel] {
  const client = new OpenAI({ apiKey, baseURL: BASE_URL });
  return [new KimiTextModel(client), new FluxImageModel(apiKey)];
}
