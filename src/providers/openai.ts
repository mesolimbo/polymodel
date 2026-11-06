import OpenAI, { toFile } from 'openai';
import { sniffImage, toDataUrl } from '../domain/imageInfo.js';
import {
  GifRequest,
  ImageModel,
  ImageRequest,
  ImageResult,
  InputImage,
  ModelInfo,
  TextModel,
  TextRequest,
  TextResult,
} from '../domain/types.js';

const TEXT_MODEL = 'gpt-6-astra';
const IMAGE_MODEL = 'gpt-image-2.6-sunburst';
const MAX_TEXT_IMAGES = 1500;
const MAX_REFERENCE_IMAGES = 16;

// gpt-image-2.6 size constraints: edges are multiples of 16, max edge 3840,
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

function toReferenceFiles(images: InputImage[]) {
  return Promise.all(
    images.map((image, i) =>
      toFile(image.bytes, `reference-${i + 1}.${image.mimeType.split('/')[1]}`, { type: image.mimeType })
    )
  );
}

export class OpenAITextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'gpt',
    provider: 'openai',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `OpenAI ${TEXT_MODEL} via the Responses API; accepts images`,
    maxInputImages: MAX_TEXT_IMAGES,
  };

  constructor(private client: OpenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const response = await this.client.responses.create({
      model: TEXT_MODEL,
      input: req.images?.length
        ? [
            {
              role: 'user',
              content: [
                ...req.images.map((image) => ({
                  type: 'input_image' as const,
                  image_url: toDataUrl(image),
                  detail: 'auto' as const,
                })),
                { type: 'input_text' as const, text: req.prompt },
              ],
            },
          ]
        : req.prompt,
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
    description: `OpenAI ${IMAGE_MODEL}; edges snap to multiples of 16, minimum ~810x810 total pixels; up to ${MAX_REFERENCE_IMAGES} reference images; supports animated GIFs`,
    maxInputImages: MAX_REFERENCE_IMAGES,
  };

  constructor(private client: OpenAI) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const { width, height } = normalizeSize(req.width, req.height);
    const size = `${width}x${height}`;
    const result = req.referenceImages?.length
      ? await this.client.images.edit({
          model: IMAGE_MODEL,
          image: await toReferenceFiles(req.referenceImages),
          prompt: req.prompt,
          size,
          output_format: 'png',
        })
      : await this.client.images.generate({
          model: IMAGE_MODEL,
          prompt: req.prompt,
          size,
          output_format: 'png',
          moderation: 'low',
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
          ? `requested ${req.width}x${req.height}, adjusted to ${width}x${height} (gpt-image-2.6 size constraints)`
          : undefined,
    };
  }

  async generateGif(req: GifRequest): Promise<ImageResult> {
    const snap = (v: number) => Math.ceil(v / EDGE_STEP) * EDGE_STEP;
    const frame = req.width && req.height ? { width: snap(req.width), height: snap(req.height) } : undefined;
    const params = {
      model: IMAGE_MODEL,
      prompt: req.prompt,
      size: frame ? `${frame.width}x${frame.height}` : 'auto',
      output_format: 'gif',
      frame_count: req.frameCount,
      frame_duration_ms: req.frameDurationMs,
    };
    // The casts cover GIF fields that the SDK's types don't include yet.
    const result = req.referenceImages?.length
      ? await this.client.images.edit({
          ...params,
          image: await toReferenceFiles(req.referenceImages),
        } as unknown as OpenAI.Images.ImageEditParamsNonStreaming)
      : await this.client.images.generate({
          ...params,
          moderation: 'low',
        } as unknown as OpenAI.Images.ImageGenerateParamsNonStreaming);
    const base64 = result.data?.[0]?.b64_json;
    const sniffed = base64 ? sniffImage(Buffer.from(base64, 'base64')) : undefined;
    if (!base64 || sniffed?.extension !== 'gif') throw new Error('OpenAI returned no GIF data');
    return {
      base64,
      mimeType: sniffed.mimeType,
      width: sniffed.width!,
      height: sniffed.height!,
      model: IMAGE_MODEL,
      sizeNote:
        frame && (frame.width !== req.width || frame.height !== req.height)
          ? `requested ${req.width}x${req.height}, adjusted to ${frame.width}x${frame.height} (GIF frame edges are multiples of 16)`
          : undefined,
    };
  }
}

export function createOpenAIModels(apiKey: string): [TextModel, ImageModel] {
  const client = new OpenAI({ apiKey });
  return [new OpenAITextModel(client), new OpenAIImageModel(client)];
}
