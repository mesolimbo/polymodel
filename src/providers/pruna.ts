import { nearestAspectRatio } from '../domain/imageInfo.js';
import { ImageModel, ImageRequest, ImageResult, InputImage, ModelInfo } from '../domain/types.js';

const BASE_URL = 'https://api.pruna.ai/v1';
const IMAGE_MODEL = 'p-image';
const EDIT_MODEL = 'p-image-edit';
const MAX_REFERENCE_IMAGES = 5;
const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 5 * 60_000;

// P-Image takes custom sizes: edges are multiples of 16 between 256 and 1440.
const EDGE_STEP = 16;
const MIN_EDGE = 256;
const MAX_EDGE = 1440;
// P-Image-Edit rejects "custom" despite the docs, so edits only get presets.
const EDIT_ASPECT_RATIOS = ['1:1', '16:9', '9:16', '4:3', '3:4', '3:2', '2:3'];

function normalizeSize(width: number, height: number): { width: number; height: number } {
  // Scale proportionally into range so the aspect ratio survives.
  let scale = Math.max(1, MIN_EDGE / Math.min(width, height));
  scale = Math.min(scale, MAX_EDGE / Math.max(width, height));
  const snap = (v: number) =>
    Math.min(MAX_EDGE, Math.max(MIN_EDGE, Math.round((v * scale) / EDGE_STEP) * EDGE_STEP));
  return { width: snap(width), height: snap(height) };
}

interface Prediction {
  status?: string;
  message?: string;
  generation_url?: string;
  get_url?: string;
}

export class PrunaImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'p-image',
    provider: 'pruna',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `Pruna P-Image, switching to P-Image-Edit when given reference images (up to ${MAX_REFERENCE_IMAGES}); edges snap to multiples of 16 between 256 and 1440, edits to the nearest preset aspect ratio`,
    maxInputImages: MAX_REFERENCE_IMAGES,
  };

  constructor(private apiKey: string) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    if (req.referenceImages?.length) return this.edit(req, req.referenceImages);

    const { width, height } = normalizeSize(req.width, req.height);
    const base64 = await this.predict(IMAGE_MODEL, { prompt: req.prompt, aspect_ratio: 'custom', width, height, disable_safety_checker: true });
    return {
      base64,
      mimeType: 'image/jpeg',
      width,
      height,
      model: IMAGE_MODEL,
      sizeNote:
        width !== req.width || height !== req.height
          ? `requested ${req.width}x${req.height}, adjusted to ${width}x${height} (P-Image size constraints)`
          : undefined,
    };
  }

  private async edit(req: ImageRequest, references: InputImage[]): Promise<ImageResult> {
    const aspectRatio = nearestAspectRatio(req.width, req.height, EDIT_ASPECT_RATIOS);
    // Inline base64 is rejected despite the docs; uploaded file URLs work.
    const images = await Promise.all(references.map((image, i) => this.upload(image, i)));
    const base64 = await this.predict(EDIT_MODEL, { prompt: req.prompt, images, aspect_ratio: aspectRatio, disable_safety_checker: true });
    return {
      base64,
      mimeType: 'image/jpeg',
      width: req.width,
      height: req.height,
      model: EDIT_MODEL,
      sizeNote: `requested ${req.width}x${req.height}, edited at preset ${aspectRatio} (P-Image-Edit supports only presets)`,
    };
  }

  private async upload(image: InputImage, index: number): Promise<string> {
    const form = new FormData();
    form.append(
      'content',
      new Blob([new Uint8Array(image.bytes)], { type: image.mimeType }),
      `reference-${index + 1}.${image.mimeType.split('/')[1]}`
    );
    const response = await fetch(`${BASE_URL}/files`, { method: 'POST', headers: { apikey: this.apiKey }, body: form });
    if (!response.ok) throw new Error(`Pruna upload error ${response.status}: ${await response.text()}`);
    const url = ((await response.json()) as { urls?: { get?: string } }).urls?.get;
    if (!url) throw new Error('Pruna upload returned no file URL');
    return url;
  }

  /** Run a prediction and return the image as base64, polling when it outlasts the 60-second sync window. */
  private async predict(model: string, input: Record<string, unknown>): Promise<string> {
    const response = await fetch(`${BASE_URL}/predictions`, {
      method: 'POST',
      headers: { apikey: this.apiKey, Model: model, 'Try-Sync': 'true', 'Content-Type': 'application/json' },
      body: JSON.stringify({ input }),
    });
    if (!response.ok) throw new Error(`Pruna API error ${response.status}: ${await response.text()}`);

    let prediction = (await response.json()) as Prediction;
    const getUrl = prediction.get_url;
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    while (!prediction.generation_url && getUrl && prediction.status !== 'failed') {
      if (Date.now() > deadline) throw new Error('Pruna prediction timed out');
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const poll = await fetch(getUrl, { headers: { apikey: this.apiKey } });
      if (!poll.ok) throw new Error(`Pruna status error ${poll.status}: ${await poll.text()}`);
      prediction = (await poll.json()) as Prediction;
    }
    if (!prediction.generation_url) {
      throw new Error(`Pruna prediction ${prediction.status ?? 'failed'}: ${prediction.message ?? JSON.stringify(prediction)}`);
    }

    const download = await fetch(prediction.generation_url, { headers: { apikey: this.apiKey } });
    if (!download.ok) throw new Error(`Failed to download image: ${download.status}`);
    return Buffer.from(await download.arrayBuffer()).toString('base64');
  }
}

export function createPrunaModels(apiKey: string): [ImageModel] {
  return [new PrunaImageModel(apiKey)];
}
