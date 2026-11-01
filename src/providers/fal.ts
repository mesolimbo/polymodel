import { nearestAspectRatio, sniffImage, toDataUrl } from '../domain/imageInfo.js';
import { ImageModel, ImageRequest, ImageResult, ModelInfo } from '../domain/types.js';

const QUEUE_URL = 'https://queue.fal.run';
const IMAGE_MODEL = 'blackforestlabs/flux-3/text-to-image';
const EDIT_MODEL = 'blackforestlabs/flux-3/edit-image';
const MAX_REFERENCE_IMAGES = 10;
const MAX_SAFETY_TOLERANCE = 4;
const POLL_INTERVAL_MS = 1000;
// 4k renders can take several minutes.
const POLL_TIMEOUT_MS = 10 * 60_000;

const ASPECT_RATIOS = ['21:9', '2:1', '16:9', '3:2', '7:5', '4:3', '5:4', '1:1', '4:5', '3:4', '5:7', '2:3', '9:16', '1:2'];

// Each tier holds about edge² pixels spread over the chosen aspect ratio,
// so 1.5k at 16:9 renders 2048x1152.
const RESOLUTION_TIERS: Array<[string, number]> = [
  ['1k', 1024],
  ['1.5k', 1536],
  ['2k', 2048],
  ['4k', 4096],
];

/** FLUX 3 takes a resolution tier rather than pixels; pick the smallest tier covering the requested area. */
function resolutionTier(width: number, height: number, aspectRatio: string): string {
  const area = width * height;
  if (aspectRatio === '1:1' && area <= 768 * 768) return '768sq';
  return RESOLUTION_TIERS.find(([, edge]) => area <= edge * edge)?.[0] ?? '4k';
}

interface QueueStatus {
  status?: string;
  response_url?: string;
  status_url?: string;
}

export class FluxImageModel implements ImageModel {
  readonly info: ModelInfo = {
    alias: 'flux-3',
    provider: 'fal',
    modality: 'image',
    underlyingModel: IMAGE_MODEL,
    description: `FLUX 3 Image served by fal, switching to FLUX 3 Edit when given reference images (up to ${MAX_REFERENCE_IMAGES}); sizes snap to the nearest preset aspect ratio and resolution tier (768sq, 1k, 1.5k, 2k, 4k)`,
    maxInputImages: MAX_REFERENCE_IMAGES,
  };

  constructor(private apiKey: string) {}

  async generateImage(req: ImageRequest): Promise<ImageResult> {
    const aspectRatio = nearestAspectRatio(req.width, req.height, ASPECT_RATIOS);
    const resolution = resolutionTier(req.width, req.height, aspectRatio);
    const model = req.referenceImages?.length ? EDIT_MODEL : IMAGE_MODEL;
    const bytes = await this.run(model, {
      prompt: req.prompt,
      aspect_ratio: aspectRatio,
      resolution,
      output_format: 'png',
      safety_tolerance: MAX_SAFETY_TOLERANCE,
      ...(req.referenceImages?.length ? { image_urls: req.referenceImages.map(toDataUrl) } : {}),
    });

    const sniffed = sniffImage(bytes);
    const width = sniffed?.width ?? req.width;
    const height = sniffed?.height ?? req.height;
    return {
      base64: bytes.toString('base64'),
      mimeType: sniffed?.mimeType ?? 'image/png',
      width,
      height,
      model,
      sizeNote:
        width !== req.width || height !== req.height
          ? `requested ${req.width}x${req.height}, rendered at ${aspectRatio} ${resolution} as ${width}x${height} (FLUX 3 supports only preset sizes)`
          : undefined,
    };
  }

  /** Submit to fal's queue, poll until done, and download the first image. */
  private async run(model: string, input: Record<string, unknown>): Promise<Buffer> {
    const headers = { Authorization: `Key ${this.apiKey}`, 'Content-Type': 'application/json' };
    const submit = await fetch(`${QUEUE_URL}/${model}`, { method: 'POST', headers, body: JSON.stringify(input) });
    if (!submit.ok) throw new Error(`fal API error ${submit.status}: ${await submit.text()}`);

    let job = (await submit.json()) as QueueStatus;
    const { status_url: statusUrl, response_url: responseUrl } = job;
    if (!statusUrl || !responseUrl) throw new Error(`fal returned no queue URLs: ${JSON.stringify(job)}`);

    const deadline = Date.now() + POLL_TIMEOUT_MS;
    while (job.status !== 'COMPLETED') {
      if (Date.now() > deadline) throw new Error('fal request timed out');
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const poll = await fetch(statusUrl, { headers });
      if (!poll.ok) throw new Error(`fal status error ${poll.status}: ${await poll.text()}`);
      job = (await poll.json()) as QueueStatus;
    }

    // A failed job still reports COMPLETED; the error surfaces on the result fetch.
    const result = await fetch(responseUrl, { headers });
    if (!result.ok) throw new Error(`fal request failed ${result.status}: ${await result.text()}`);
    const url = ((await result.json()) as { images?: Array<{ url?: string }> }).images?.[0]?.url;
    if (!url) throw new Error('fal returned no image');

    if (url.startsWith('data:')) return Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');
    const download = await fetch(url);
    if (!download.ok) throw new Error(`Failed to download image: ${download.status}`);
    return Buffer.from(await download.arrayBuffer());
  }
}

export function createFalModels(apiKey: string): [ImageModel] {
  return [new FluxImageModel(apiKey)];
}
