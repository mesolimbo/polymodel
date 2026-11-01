import OpenAI from 'openai';
import type { ChatCompletionCreateParams } from 'openai/resources/chat/completions';
import { toDataUrl } from '../domain/imageInfo.js';
import { ModelInfo, TextModel, TextRequest, TextResult } from '../domain/types.js';

const BASE_URL = 'https://api.together.xyz/v1';
const TEXT_MODEL = 'moonshotai/Kimi-K3';
// Together caps the whole request body at 100 MB rather than the image count.
const MAX_TEXT_IMAGES = 1000;

// K3 always reasons, and its thinking counts against max_completion_tokens, so a
// budget sized for the answer alone is spent before any content is emitted.
// max_tokens is therefore treated as the answer budget and reasoning headroom is
// added on top. An unused ceiling costs nothing; only generated tokens bill.
const DEFAULT_ANSWER_TOKENS = 32_768;
const MAX_COMPLETION_TOKENS = 1_048_576;
const REASONING_HEADROOM: Record<string, number> = {
  none: 0,
  low: 16_384,
  medium: 49_152,
  high: 65_536,
  xhigh: 98_304,
  max: 98_304,
};
// Together leaves K3 at max effort when the caller says nothing.
const DEFAULT_HEADROOM = REASONING_HEADROOM.max;

function completionBudget(req: TextRequest): number {
  const headroom = (req.reasoningEffort ? REASONING_HEADROOM[req.reasoningEffort] : undefined) ?? DEFAULT_HEADROOM;
  return Math.min(MAX_COMPLETION_TOKENS, (req.maxTokens ?? DEFAULT_ANSWER_TOKENS) + headroom);
}

export class KimiTextModel implements TextModel {
  readonly info: ModelInfo = {
    alias: 'kimi-k3',
    provider: 'together',
    modality: 'text',
    underlyingModel: TEXT_MODEL,
    description: `Moonshot Kimi K3 served by Together AI; always reasons, so its reasoning allowance is budgeted on top of max_tokens; accepts images`,
    maxInputImages: MAX_TEXT_IMAGES,
  };

  constructor(private client: OpenAI) {}

  async generateText(req: TextRequest): Promise<TextResult> {
    const budget = completionBudget(req);
    // Streamed so bytes keep flowing: Together drops the connection on long
    // non-streaming generations (429 worker_stream_failed after ~10 minutes).
    const stream = await this.client.chat.completions.create({
      model: TEXT_MODEL,
      messages: [
        {
          role: 'user',
          content: req.images?.length
            ? [
                ...req.images.map((image) => ({
                  type: 'image_url' as const,
                  image_url: { url: toDataUrl(image) },
                })),
                { type: 'text' as const, text: req.prompt },
              ]
            : req.prompt,
        },
      ],
      max_completion_tokens: budget,
      temperature: req.temperature,
      ...(req.reasoningEffort
        ? { reasoning_effort: req.reasoningEffort as ChatCompletionCreateParams['reasoning_effort'] }
        : {}),
      stream: true,
    });
    let text = '';
    let finishReason: string | null | undefined;
    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      text += choice?.delta?.content ?? '';
      finishReason = choice?.finish_reason ?? finishReason;
    }
    if (finishReason === 'length') {
      const advice = 'Raise max_tokens or lower reasoning_effort.';
      text = text
        ? `${text}\n\n[Output truncated at the ${budget}-token budget; reasoning counts toward it. ${advice}]`
        : `[No answer: the ${budget}-token budget ran out while the model was still reasoning. ${advice}]`;
    }
    return { text: text || 'No response received', model: TEXT_MODEL };
  }
}

export function createTogetherModels(apiKey: string): [TextModel] {
  const client = new OpenAI({ apiKey, baseURL: BASE_URL });
  return [new KimiTextModel(client)];
}
