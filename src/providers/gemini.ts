import { FinishReason, GoogleGenAI, ThinkingLevel } from '@google/genai';
import {
  ImageModel,
  ImageRequest,
  ImageResult,
  ModelInfo,
  TextModel,
  TextRequest,
  TextResult,
} from '../domain/types.js';

const TEXT_MODEL = 'gemini-pro-latest';
const IMAGE_MODEL = 'gemini-3-pro-image';

// Gemini 3.x always thinks and its thinking counts against maxOutputTokens, so a
// budget sized for the answer alone is spent before the answer starts. max_tokens
// is treated as the answer budget and thinking headroom is added on top; an unused
// ceiling costs nothing, since only generated tokens bill.
const MAX_OUTPUT_TOKENS = 65_536;
const DEFAULT_ANSWER_TOKENS = 16_384;
// The model rejects any level outside low/medium/high and refuses to stop thinking
// at all, so the shared effort scale is mapped onto what it accepts.
const THINKING_LEVELS: Record<string, ThinkingLevel> = {
  none: ThinkingLevel.LOW,
  low: ThinkingLevel.LOW,
  medium: ThinkingLevel.MEDIUM,
  high: ThinkingLevel.HIGH,
  xhigh: ThinkingLevel.HIGH,
  max: ThinkingLevel.HIGH,
};
const THINKING_HEADROOM: Record<ThinkingLevel, number> = {
  [ThinkingLevel.THINKING_LEVEL_UNSPECIFIED]: 32_768,
  [ThinkingLevel.MINIMAL]: 4_096,
  [ThinkingLevel.LOW]: 8_192,
  [ThinkingLevel.MEDIUM]: 16_384,
  [ThinkingLevel.HIGH]: 32_768,
};
// Left to itself the model decides how long to think, so it gets the deepest allowance.
const DEFAULT_HEADROOM = THINKING_HEADROOM[ThinkingLevel.HIGH];

// Nano Banana Pro only takes preset aspect ratios and 1K/2K/4K sizes, so the
// requested pixel dimensions are mapped to the nearest supported combination.
const ASPECT_RATIOS: Array<[string, number]> = [
  ['1:1', 1],
  ['2:3', 2 / 3],
  ['3:2', 3 / 2],
  ['3:4', 3 / 4],
  ['4:3', 4 / 3],
  ['9:16', 9 / 16],
  ['16:9', 16 / 9],
  ['21:9', 21 / 9],
];

function nearestAspectRatio(width: number, height: number): string {
  const target = width / height;
  let best = ASPECT_RATIOS[0];
  for (const candidate of ASPECT_RATIOS) {
    if (Math.abs(Math.log(candidate[1] / target)) < Math.abs(Math.log(best[1] / target))) {
      best = candidate;
    }
  }
  return best[0];
}

function nearestImageSize(width: number, height: number): '1K' | '2K' | '4K' {
  const edge = Math.max(width, height);
  if (edge <= 1024) return '1K';
  if (edge <= 2048) return '2K';
  return '4K';
}

export class GeminiTextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'gemini',
    provider: 'gemini',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `Google ${TEXT_MODEL} (currently gemini-3.1-pro-preview); always thinks, so its thinking allowance is budgeted on top of max_tokens`,
  };

  constructor(private client: GoogleGenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const thinkingLevel = req.reasoningEffort ? THINKING_LEVELS[req.reasoningEffort] : undefined;
    const headroom = thinkingLevel ? THINKING_HEADROOM[thinkingLevel] : DEFAULT_HEADROOM;
    const budget = Math.min(MAX_OUTPUT_TOKENS, (req.maxTokens ?? DEFAULT_ANSWER_TOKENS) + headroom);
    const response = await this.client.models.generateContent({
      model: TEXT_MODEL,
      contents: req.prompt,
      config: {
        maxOutputTokens: budget,
        temperature: req.temperature,
        ...(thinkingLevel ? { thinkingConfig: { thinkingLevel } } : {}),
      },
    });
    let text = response.text ?? '';
    if (response.candidates?.[0]?.finishReason === FinishReason.MAX_TOKENS) {
      const advice = 'Raise max_tokens or lower reasoning_effort.';
      text = text
        ? `${text}\n\n[Output truncated at the ${budget}-token budget; thinking counts toward it. ${advice}]`
        : `[No answer: the ${budget}-token budget ran out while the model was still thinking. ${advice}]`;
    }
    return { text: text || 'No response received', model: TEXT_MODEL };
  }
}

export class GeminiImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'nano-banana',
    provider: 'gemini',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `Google ${IMAGE_MODEL} (Nano Banana Pro); size maps to nearest preset (1K/2K/4K + aspect ratio)`,
  };

  constructor(private client: GoogleGenAI) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const aspectRatio = nearestAspectRatio(req.width, req.height);
    const imageSize = nearestImageSize(req.width, req.height);
    const response = await this.client.models.generateContent({
      model: IMAGE_MODEL,
      contents: req.prompt,
      config: {
        responseModalities: ['IMAGE'],
        imageConfig: { aspectRatio, imageSize },
      },
    });

    const parts = response.candidates?.[0]?.content?.parts ?? [];
    for (const part of parts) {
      if (part.inlineData?.data) {
        return {
          base64: part.inlineData.data,
          mimeType: part.inlineData.mimeType ?? 'image/png',
          width: req.width,
          height: req.height,
          model: IMAGE_MODEL,
          sizeNote: `requested ${req.width}x${req.height}, generated with preset ${imageSize} at ${aspectRatio} (Gemini supports only presets)`,
        };
      }
    }
    throw new Error('Gemini returned no image data');
  }
}

export function createGeminiModels(apiKey: string): [TextModel, ImageModel] {
  const client = new GoogleGenAI({ apiKey });
  return [new GeminiTextModel(client), new GeminiImageModel(client)];
}
