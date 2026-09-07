import OpenAI from 'openai';
import type { ChatCompletionCreateParams } from 'openai/resources/chat/completions';
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
const IMAGE_MODEL = 'black-forest-labs/FLUX.2-max';

// K3 always reasons, and its thinking counts against max_completion_tokens, so a
// budget sized for the answer alone is spent before any content is emitted.
// max_tokens is therefore treated as the answer budget and reasoning headroom is
// added on top. An unused ceiling costs nothing; only generated tokens bill.
const DEFAULT_ANSWER_TOKENS = 32_768;
const MAX_COMPLETION_TOKENS = 1_048_576;
const REASONING_HEADROOM: Record<string, number> = {
  none: 0,
  low: 16_384,
  medium: 49_152,
  high: 65_536,
  xhigh: 98_304,
  max: 98_304,
};
// Together leaves K3 at max effort when the caller says nothing.
const DEFAULT_HEADROOM = REASONING_HEADROOM.max;

function completionBudget(req: TextRequest): number {
  const headroom = (req.reasoningEffort ? REASONING_HEADROOM[req.reasoningEffort] : undefined) ?? DEFAULT_HEADROOM;
  return Math.min(MAX_COMPLETION_TOKENS, (req.maxTokens ?? DEFAULT_ANSWER_TOKENS) + headroom);
}

// FLUX.2 max accepts direct pixel dimensions; snap to multiples of 16 within
// the 256-2048 range Together serves.
const EDGE_STEP = 16;
const MIN_EDGE = 256;
const MAX_EDGE = 2048;

function normalizeSize(width: number, height: number): { width: number; height: number } {
  // Scale proportionally into range so the aspect ratio survives.
  let scale = Math.max(1, MIN_EDGE / Math.min(width, height));
  scale = Math.min(scale, MAX_EDGE / Math.max(width, height));
  const snap = (v: number) =>
    Math.min(MAX_EDGE, Math.max(MIN_EDGE, Math.round((v * scale) / EDGE_STEP) * EDGE_STEP));
  return { width: snap(width), height: snap(height) };
}

export class KimiTextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'kimi-k3',
    provider: 'together',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `Moonshot Kimi K3 served by Together AI; always reasons, so its reasoning allowance is budgeted on top of max_tokens`,
  };

  constructor(private client: OpenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const budget = completionBudget(req);
    // Streamed so bytes keep flowing: Together drops the connection on long
    // non-streaming generations (429 worker_stream_failed after ~10 minutes).
    const stream = await this.client.chat.completions.create({
      model: TEXT_MODEL,
      messages: [{ role: 'user', content: req.prompt }],
      max_completion_tokens: budget,
      temperature: req.temperature,
      ...(req.reasoningEffort
        ? { reasoning_effort: req.reasoningEffort as ChatCompletionCreateParams['reasoning_effort'] }
        : {}),
      stream: true,
    });
    let text = '';
    let finishReason: string | null | undefined;
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      text += choice?.delta?.content ?? '';
      finishReason = choice?.finish_reason ?? finishReason;
    }
    if (finishReason === 'length') {
      const advice = 'Raise max_tokens or lower reasoning_effort.';
      text = text
        ? `${text}\n\n[Output truncated at the ${budget}-token budget; reasoning counts toward it. ${advice}]`
        : `[No answer: the ${budget}-token budget ran out while the model was still reasoning. ${advice}]`;
    }
    return { text: text || 'No response received', model: TEXT_MODEL };
  }
}

export class FluxImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'flux-2-max',
    provider: 'together',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `FLUX.2 [max] served by Together AI; edges snap to multiples of 16 between 256 and 2048`,
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
          ? `requested ${req.width}x${req.height}, adjusted to ${width}x${height} (FLUX.2 max size constraints)`
          : undefined,
    };
  }
}

export function createTogetherModels(apiKey: string): [TextModel, ImageModel] {
  const client = new OpenAI({ apiKey, baseURL: BASE_URL });
  return [new KimiTextModel(client), new FluxImageModel(apiKey)];
}
