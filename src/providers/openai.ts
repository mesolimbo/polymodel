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

const TEXT_MODEL = 'gpt-6-astra';
const IMAGE_MODEL = 'gpt-image-2';

// gpt-image-2 size constraints: edges are multiples of 16, max edge 3840,
// total pixels between 655,360 and 8,294,400.
const EDGE_STEP = 16;
const MAX_EDGE = 3840;
const MIN_PIXELS = 655_360;
const MAX_PIXELS = 8_294_400;

function normalizeSize(width: number, height: number): { width: number; height: number } {
  const scaleFor = (w: number, h: number) => {
    const pixels = w * h;
    if (pixels < MIN_PIXELS) return Math.sqrt(MIN_PIXELS / pixels);
    if (pixels > MAX_PIXELS) return Math.sqrt(MAX_PIXELS / pixels);
    return 1;
  };
  const scale = scaleFor(width, height);
  const snap = (v: number) =>
    Math.min(MAX_EDGE, Math.max(EDGE_STEP, Math.ceil((v * scale) / EDGE_STEP) * EDGE_STEP));
  return { width: snap(width), height: snap(height) };
}

export class OpenAITextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'gpt',
    provider: 'openai',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `OpenAI ${TEXT_MODEL} via the Responses API`,
  };

  constructor(private client: OpenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const response = await this.client.responses.create({
      model: TEXT_MODEL,
      input: req.prompt,
      reasoning: { effort: (req.reasoningEffort ?? 'low') as 'low' },
      max_output_tokens: req.maxTokens ?? 16384,
    });
    return { text: response.output_text || 'No response received', model: TEXT_MODEL };
  }
}

export class OpenAIImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'gpt-image',
    provider: 'openai',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `OpenAI ${IMAGE_MODEL}; edges snap to multiples of 16, minimum ~810x810 total pixels`,
  };

  constructor(private client: OpenAI) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const { width, height } = normalizeSize(req.width, req.height);
    const result = await this.client.images.generate({
      model: IMAGE_MODEL,
      prompt: req.prompt,
      size: `${width}x${height}`,
      output_format: 'png',
    });
    const base64 = result.data?.[0]?.b64_json;
    if (!base64) throw new Error('OpenAI returned no image data');
    return {
      base64,
      mimeType: 'image/png',
      width,
      height,
      model: IMAGE_MODEL,
      sizeNote:
        width !== req.width || height !== req.height
          ? `requested ${req.width}x${req.height}, adjusted to ${width}x${height} (gpt-image-2 size constraints)`
          : undefined,
    };
  }
}

export function createOpenAIModels(apiKey: string): [TextModel, ImageModel] {
  const client = new OpenAI({ apiKey });
  return [new OpenAITextModel(client), new OpenAIImageModel(client)];
}
