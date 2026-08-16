import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface Config {
  openaiApiKey?: string;
  geminiApiKey?: string;
  togetherApiKey?: string;
}

/**
 * Keys come from the environment; a .env file in the project root is loaded
 * first but never overrides variables already set in the environment.
 */
export function loadConfig(): Config {
  const envPath = join(__dirname, '..', '.env');
  if (existsSync(envPath)) {
    const before = { ...process.env };
    process.loadEnvFile(envPath);
    for (const [key, value] of Object.entries(before)) {
      if (value !== undefined) process.env[key] = value;
    }
  }

  return {
    openaiApiKey: process.env.OPENAI_API_KEY,
    geminiApiKey: process.env.GEMINI_API_KEY,
    togetherApiKey: process.env.TOGETHER_API_KEY,
  };
}
