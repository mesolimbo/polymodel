export type Modality = 'text' | 'image' | 'sound' | 'music';

export interface ModelInfo {
  /** Stable alias callers use to select the model, e.g. "flux-2-dev". */
  alias: string;
  provider: string;
  modality: Modality;
  /** Provider-side model identifier, e.g. "black-forest-labs/FLUX.2-dev". */
  underlyingModel: string;
  description: string;
  /** Images the model accepts per request; unset when it takes none. */
  maxInputImages?: number;
}

export interface InputImage {
  bytes: Buffer;
  mimeType: string;
}

export interface TextRequest {
  prompt: string;
  images?: InputImage[];
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
  /** Images the model should draw on, in the order the prompt refers to them. */
  referenceImages?: InputImage[];
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

export interface AudioFile {
  bytes: Buffer;
  mimeType: string;
  extension: string;
}

export interface SoundRequest {
  prompt: string;
  durationSeconds?: number;
  loop?: boolean;
  promptInfluence?: number;
  outputFormat?: string;
}

export interface SoundResult {
  audio: AudioFile;
  model: string;
  /** Provider billing units consumed, when reported. */
  cost?: string;
}

/** Provider-defined plan structure; passed through untouched. */
export type CompositionPlan = Record<string, unknown>;

export interface MusicRequest {
  prompt?: string;
  compositionPlan?: CompositionPlan;
  lengthMs?: number;
  generationMode?: string;
  forceInstrumental?: boolean;
  seed?: number;
  finetuneId?: string;
  finetuneStrength?: number;
  usePhoneticNames?: boolean;
  respectSectionsDurations?: boolean;
  storeForInpainting?: boolean;
  withTimestamps?: boolean;
  withWaveformVisual?: boolean;
  signWithC2pa?: boolean;
  outputFormat?: string;
}

export interface MusicResult {
  audio: AudioFile;
  model: string;
  /** Id for referencing this song in later plans (inpainting, conditioning). */
  songId?: string;
  /** Plan, metadata, timestamps and waveform the provider returned alongside the audio. */
  details?: Record<string, unknown>;
}

export interface MusicPlanRequest {
  prompt: string;
  lengthMs?: number;
  sourcePlan?: CompositionPlan;
}

export interface VideoToMusicRequest {
  videoPaths: string[];
  description?: string;
  tags?: string[];
  signWithC2pa?: boolean;
  outputFormat?: string;
}

export interface TextModel {
  readonly info: ModelInfo;
  generateText(req: TextRequest): Promise<TextResult>;
}

export interface ImageModel {
  readonly info: ModelInfo;
  generateImage(req: ImageRequest): Promise<ImageResult>;
}

export interface SoundModel {
  readonly info: ModelInfo;
  generateSound(req: SoundRequest): Promise<SoundResult>;
}

export interface MusicModel {
  readonly info: ModelInfo;
  compose(req: MusicRequest): Promise<MusicResult>;
  createPlan(req: MusicPlanRequest): Promise<CompositionPlan>;
  /** Returns a single track, or a ZIP archive when the provider sends several. */
  videoToMusic(req: VideoToMusicRequest): Promise<AudioFile>;
}

export type AnyModel = TextModel | ImageModel | SoundModel | MusicModel;

export function isTextModel(m: AnyModel): m is TextModel {
  return m.info.modality === 'text';
}

export function isImageModel(m: AnyModel): m is ImageModel {
  return m.info.modality === 'image';
}

export function isSoundModel(m: AnyModel): m is SoundModel {
  return m.info.modality === 'sound';
}

export function isMusicModel(m: AnyModel): m is MusicModel {
  return m.info.modality === 'music';
}
