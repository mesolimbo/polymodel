import { FinishReason, GoogleGenAI, HarmBlockThreshold, HarmCategory, Part, SafetySetting, ThinkingLevel } from '@google/genai';
import { nearestAspectRatio } from '../domain/imageInfo.js';
import {
  ImageModel,
  ImageRequest,
  ImageResult,
  InputImage,
  ModelInfo,
  TextModel,
  TextRequest,
  TextResult,
} from '../domain/types.js';

const TEXT_MODEL = 'gemini-4-argon';
const IMAGE_MODEL = 'gemini-3-pro-image';
const MAX_TEXT_IMAGES = 3600;
const MAX_REFERENCE_IMAGES = 14;

// Gemini 4 Argon always thinks and its thinking counts against maxOutputTokens, so a
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
// Argon thinks at medium unless told otherwise.
const DEFAULT_HEADROOM = THINKING_HEADROOM[ThinkingLevel.MEDIUM];

// Nano Banana Pro only takes preset aspect ratios and 1K/2K/4K sizes, so the
// requested pixel dimensions are mapped to the nearest supported combination.
const ASPECT_RATIOS = ['1:1', '2:3', '3:2', '3:4', '4:3', '9:16', '16:9', '21:9'];

// Every adjustable filter turned off; Google still blocks core harms such as child safety.
const SAFETY_SETTINGS: SafetySetting[] = [
  HarmCategory.HARM_CATEGORY_HARASSMENT,
  HarmCategory.HARM_CATEGORY_HATE_SPEECH,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
  HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
].map((category) => ({ category, threshold: HarmBlockThreshold.OFF }));

function promptParts(prompt: string, images: InputImage[] = []): Part[] {
  return [
    ...images.map((image) => ({ inlineData: { data: image.bytes.toString('base64'), mimeType: image.mimeType } })),
    { text: prompt },
  ];
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
    description: `Google ${TEXT_MODEL}; always thinks, so its thinking allowance is budgeted on top of max_tokens; accepts images`,
    maxInputImages: MAX_TEXT_IMAGES,
  };

  constructor(private client: GoogleGenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const thinkingLevel = req.reasoningEffort ? THINKING_LEVELS[req.reasoningEffort] : undefined;
    const headroom = thinkingLevel ? THINKING_HEADROOM[thinkingLevel] : DEFAULT_HEADROOM;
    const budget = Math.min(MAX_OUTPUT_TOKENS, (req.maxTokens ?? DEFAULT_ANSWER_TOKENS) + headroom);
    const response = await this.client.models.generateContent({
      model: TEXT_MODEL,
      contents: promptParts(req.prompt, req.images),
      config: {
        maxOutputTokens: budget,
        safetySettings: SAFETY_SETTINGS,
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
    description: `Google ${IMAGE_MODEL} (Nano Banana Pro); size maps to nearest preset (1K/2K/4K + aspect ratio); up to ${MAX_REFERENCE_IMAGES} reference images`,
    maxInputImages: MAX_REFERENCE_IMAGES,
  };

  constructor(private client: GoogleGenAI) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const aspectRatio = nearestAspectRatio(req.width, req.height, ASPECT_RATIOS);
    const imageSize = nearestImageSize(req.width, req.height);
    const response = await this.client.models.generateContent({
      model: IMAGE_MODEL,
      contents: promptParts(req.prompt, req.referenceImages),
      config: {
        responseModalities: ['IMAGE'],
        imageConfig: { aspectRatio, imageSize },
        safetySettings: SAFETY_SETTINGS,
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
