export type Modality = 'text' | 'image';

export interface ModelInfo {
  /** Stable alias callers use to select the model, e.g. "flux-2-dev". */
  alias: string;
  provider: string;
  modality: Modality;
  /** Provider-side model identifier, e.g. "black-forest-labs/FLUX.2-dev". */
  underlyingModel: string;
  description: string;
}

export interface TextRequest {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  reasoningEffort?: string;
}

export interface TextResult {
  text: string;
  model: string;
}

export interface ImageRequest {
  prompt: string;
  width: number;
  height: number;
}

export interface ImageResult {
  base64: string;
  mimeType: string;
  /** Actual dimensions after provider-specific normalization. */
  width: number;
  height: number;
  model: string;
  /** Set when the provider could not honor the requested size exactly. */
  sizeNote?: string;
}

export interface TextModel {
  readonly info: ModelInfo;
  generateText(req: TextRequest): Promise<TextResult>;
}

export interface ImageModel {
  readonly info: ModelInfo;
  generateImage(req: ImageRequest): Promise<ImageResult>;
}

export type AnyModel = TextModel | ImageModel;

export function isTextModel(m: AnyModel): m is TextModel {
  return m.info.modality === 'text';
}

export function isImageModel(m: AnyModel): m is ImageModel {
  return m.info.modality === 'image';
}
