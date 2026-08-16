import { GoogleGenAI } from '@google/genai';
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
const IMAGE_MODEL = 'gemini-3.1-flash-image';

// Nano Banana 2 only takes preset aspect ratios and 1K/2K/4K sizes, so the
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
    description: `Google ${TEXT_MODEL}`,
  };

  constructor(private client: GoogleGenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const response = await this.client.models.generateContent({
      model: TEXT_MODEL,
      contents: req.prompt,
      config: {
        maxOutputTokens: req.maxTokens ?? 8192,
        temperature: req.temperature,
      },
    });
    return { text: response.text || 'No response received', model: TEXT_MODEL };
  }
}

export class GeminiImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'nano-banana',
    provider: 'gemini',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `Google ${IMAGE_MODEL} (Nano Banana 2); size maps to nearest preset (1K/2K/4K + aspect ratio)`,
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
