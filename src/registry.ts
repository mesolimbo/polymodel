import {
  AnyModel,
  ImageModel,
  ModelInfo,
  TextModel,
  isImageModel,
  isTextModel,
} from './domain/types.js';

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
    const model = this.get(alias);
    if (!isTextModel(model)) {
      throw new Error(`Model "${alias}" is an image model; use generate_image instead`);
    }
    return model;
  }

  getImage(alias: string): ImageModel {
    const model = this.get(alias);
    if (!isImageModel(model)) {
      throw new Error(`Model "${alias}" is a text model; use generate_text instead`);
    }
    return model;
  }

  list(): ModelInfo[] {
    return [...this.models.values()].map((m) => m.info);
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
