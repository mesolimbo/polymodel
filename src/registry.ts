import {
  AnyModel,
  ImageModel,
  Modality,
  ModelInfo,
  MusicModel,
  SoundModel,
  TextModel,
  isImageModel,
  isMusicModel,
  isSoundModel,
  isTextModel,
} from './domain/types.js';

const TOOL_FOR: Record<Modality, string> = {
  text: 'generate_text',
  image: 'generate_image',
  sound: 'generate_sound_effect',
  music: 'generate_music',
};

/**
 * Central lookup of registered models. Providers contribute models at startup;
 * the tool layer only ever talks to this registry, never to providers directly.
 */
export class ModelRegistry {
  private models = new Map<string, AnyModel>();

  register(...models: AnyModel[]): void {
    for (const model of models) {
      const key = model.info.alias.toLowerCase();
      if (this.models.has(key)) {
        throw new Error(`Duplicate model alias: ${model.info.alias}`);
      }
      this.models.set(key, model);
    }
  }

  getText(alias: string): TextModel {
    return this.getAs(alias, 'text', isTextModel);
  }

  getImage(alias: string): ImageModel {
    return this.getAs(alias, 'image', isImageModel);
  }

  getSound(alias: string): SoundModel {
    return this.getAs(alias, 'sound', isSoundModel);
  }

  getMusic(alias: string): MusicModel {
    return this.getAs(alias, 'music', isMusicModel);
  }

  list(): ModelInfo[] {
    return [...this.models.values()].map((m) => m.info);
  }

  aliases(modality: Modality): string[] {
    return this.list()
      .filter((m) => m.modality === modality)
      .map((m) => m.alias);
  }

  private getAs<M extends AnyModel>(alias: string, modality: Modality, guard: (m: AnyModel) => m is M): M {
    const model = this.get(alias);
    if (!guard(model)) {
      const actual = model.info.modality;
      throw new Error(`Model "${alias}" is a ${actual} model; use ${TOOL_FOR[actual]} instead`);
    }
    return model;
  }

  private get(alias: string): AnyModel {
    const model = this.models.get(alias.toLowerCase());
    if (!model) {
      const known = [...this.models.keys()].join(', ') || '(none - check API keys in .env)';
      throw new Error(`Unknown model "${alias}". Available: ${known}`);
    }
    return model;
  }
}
