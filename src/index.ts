#!/usr/bin/env node

import { McpServer } from '@modelcontextprotocol/server';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import * as z from 'zod';
import { writeFileSync } from 'fs';
import { basename, resolve } from 'path';
import { sniffImage } from './domain/imageInfo.js';
import { createRequire } from 'module';
import { loadConfig } from './config.js';
import { ModelRegistry } from './registry.js';
import { createOpenAIModels } from './providers/openai.js';
import { createGeminiModels } from './providers/gemini.js';
import { createTogetherModels } from './providers/together.js';

const pkg = createRequire(import.meta.url)('../package.json') as {
  name: string;
  version: string;
};

const DEFAULT_IMAGE_EDGE = 750;

const config = loadConfig();
const registry = new ModelRegistry();

if (config.openaiApiKey) registry.register(...createOpenAIModels(config.openaiApiKey));
if (config.geminiApiKey) registry.register(...createGeminiModels(config.geminiApiKey));
if (config.togetherApiKey) registry.register(...createTogetherModels(config.togetherApiKey));

if (registry.list().length === 0) {
  console.error('No API keys found. Copy .env.example to .env and add at least one key.');
  process.exit(1);
}

const server = new McpServer({ name: pkg.name, version: pkg.version });

const textAliases = registry
  .list()
  .filter((m) => m.modality === 'text')
  .map((m) => m.alias)
  .join(', ');
const imageAliases = registry
  .list()
  .filter((m) => m.modality === 'image')
  .map((m) => m.alias)
  .join(', ');

function errorResult(prefix: string, error: unknown) {
  return {
    content: [
      {
        type: 'text' as const,
        text: `${prefix}: ${error instanceof Error ? error.message : 'Unknown error'}`,
      },
    ],
    isError: true,
  };
}

server.registerTool(
  'generate_text',
  {
    title: 'Generate text',
    description: `Query an external text model with a prompt. Available models: ${textAliases}`,
    inputSchema: z.object({
      model: z.string().describe(`Model alias to use (${textAliases})`),
      prompt: z.string().describe('The prompt to send to the model'),
      max_tokens: z.number().optional().describe('Maximum tokens in the response'),
      temperature: z.number().min(0).max(2).optional().describe('Sampling temperature'),
      reasoning_effort: z
        .enum(['none', 'low', 'medium', 'high', 'xhigh', 'max'])
        .optional()
        .describe('Reasoning effort, for models that support it'),
    }),
  },
  async ({ model, prompt, max_tokens, temperature, reasoning_effort }) => {
    try {
      const result = await registry.getText(model).generateText({
        prompt,
        maxTokens: max_tokens,
        temperature,
        reasoningEffort: reasoning_effort,
      });
      return { content: [{ type: 'text', text: result.text }] };
    } catch (error) {
      return errorResult(`Error querying ${model}`, error);
    }
  }
);

server.registerTool(
  'generate_image',
  {
    title: 'Generate image',
    description:
      `Generate an image with an external model. Available models: ${imageAliases}. ` +
      `Defaults to a ${DEFAULT_IMAGE_EDGE}x${DEFAULT_IMAGE_EDGE} square; providers snap the size to their nearest supported dimensions. ` +
      'The image is saved to disk and the file path returned.',
    inputSchema: z.object({
      model: z.string().describe(`Model alias to use (${imageAliases})`),
      prompt: z.string().describe('Description of the image to generate'),
      width: z.number().int().positive().default(DEFAULT_IMAGE_EDGE).describe('Requested width in pixels'),
      height: z.number().int().positive().default(DEFAULT_IMAGE_EDGE).describe('Requested height in pixels'),
      output_path: z
        .string()
        .optional()
        .describe('Where to save the image; defaults to polymodel-<model>-<n>.png in the current directory'),
      return_image: z
        .boolean()
        .default(false)
        .describe('Also return the image content inline (costs context; the file is written regardless)'),
    }),
  },
  async ({ model, prompt, width, height, output_path, return_image }) => {
    try {
      const result = await registry.getImage(model).generateImage({ prompt, width, height });
      const bytes = Buffer.from(result.base64, 'base64');
      const sniffed = sniffImage(bytes);
      const mimeType = sniffed?.mimeType ?? result.mimeType;
      const extension = sniffed?.extension ?? 'png';

      let path: string;
      if (output_path) {
        // Append the real extension when the caller left it off.
        path = resolve(/\.[A-Za-z0-9]+$/.test(basename(output_path)) ? output_path : `${output_path}.${extension}`);
      } else {
        path = resolve(`polymodel-${model}-${Date.now()}.${extension}`);
      }
      writeFileSync(path, bytes);

      const actualSize =
        sniffed?.width && sniffed?.height
          ? `${sniffed.width}x${sniffed.height}`
          : `${result.width}x${result.height}`;
      const summary = [
        `Saved image to ${path}`,
        `model: ${result.model}`,
        `format: ${mimeType}`,
        `size: ${actualSize}`,
        ...(result.sizeNote ? [`note: ${result.sizeNote}`] : []),
      ].join('\n');

      return {
        content: [
          { type: 'text' as const, text: summary },
          ...(return_image ? [{ type: 'image' as const, data: result.base64, mimeType }] : []),
        ],
      };
    } catch (error) {
      return errorResult(`Error generating image with ${model}`, error);
    }
  }
);

server.registerTool(
  'list_models',
  {
    title: 'List models',
    description: 'List all available models with their provider, modality, and underlying model id',
    inputSchema: z.object({}),
    outputSchema: z.object({
      models: z.array(
        z.object({
          alias: z.string(),
          provider: z.string(),
          modality: z.enum(['text', 'image']),
          underlyingModel: z.string(),
          description: z.string(),
        })
      ),
    }),
  },
  async () => {
    const output = { models: registry.list() };
    return {
      content: [{ type: 'text', text: JSON.stringify(output, null, 2) }],
      structuredContent: output,
    };
  }
);

server.registerTool(
  'get_version',
  {
    title: 'Get version',
    description: 'Get the name and version of this MCP server',
    inputSchema: z.object({}),
  },
  async () => ({
    content: [{ type: 'text', text: JSON.stringify({ name: pkg.name, version: pkg.version }, null, 2) }],
  })
);

async function main() {
  await server.connect(new StdioServerTransport());
  console.error('Polymodel MCP Server running on stdio');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
