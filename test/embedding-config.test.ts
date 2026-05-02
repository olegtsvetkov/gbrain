import { afterEach, describe, expect, test } from 'bun:test';
import {
  EMBEDDING_COST_PER_1K_TOKENS,
  embedBatch,
  isEmbeddingConfigured,
  resolveEmbeddingConfig,
} from '../src/core/embedding.ts';

const ENV_KEYS = [
  'GBRAIN_EMBEDDING_PROVIDER',
  'GBRAIN_EMBEDDING_MODEL',
  'GBRAIN_EMBEDDING_DIMENSIONS',
  'GBRAIN_EMBEDDING_BASE_URL',
  'GBRAIN_EMBEDDING_API_KEY',
  'GBRAIN_EMBED_BATCH_SIZE',
  'GBRAIN_EMBEDDING_COST_PER_1K_TOKENS',
  'OPENAI_API_KEY',
  'OPENROUTER_API_KEY',
  'OLLAMA_HOST',
] as const;

const savedEnv = new Map<string, string | undefined>();
for (const key of ENV_KEYS) savedEnv.set(key, process.env[key]);

afterEach(() => {
  for (const key of ENV_KEYS) {
    const value = savedEnv.get(key);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

describe('embedding provider config', () => {
  test('defaults to OpenAI-compatible 1536-dimensional embeddings', () => {
    delete process.env.OPENAI_API_KEY;

    const config = resolveEmbeddingConfig(null);

    expect(config.provider).toBe('openai');
    expect(config.model).toBe('text-embedding-3-large');
    expect(config.dimensions).toBe(1536);
    expect(config.costPer1kTokens).toBe(EMBEDDING_COST_PER_1K_TOKENS);
    expect(isEmbeddingConfigured(null)).toBe(false);
  });

  test('resolves OpenRouter from env', () => {
    process.env.GBRAIN_EMBEDDING_PROVIDER = 'openrouter';
    process.env.OPENROUTER_API_KEY = 'or-test';

    const config = resolveEmbeddingConfig(null);

    expect(config.provider).toBe('openrouter');
    expect(config.baseURL).toBe('https://openrouter.ai/api/v1');
    expect(config.model).toBe('openai/text-embedding-3-large');
    expect(config.apiKey).toBe('or-test');
    expect(isEmbeddingConfigured(null)).toBe(true);
  });

  test('treats Ollama as configured without an API key', () => {
    process.env.GBRAIN_EMBEDDING_PROVIDER = 'ollama';
    process.env.OLLAMA_HOST = 'http://localhost:11435';

    const config = resolveEmbeddingConfig(null);

    expect(config.provider).toBe('ollama');
    expect(config.baseURL).toBe('http://localhost:11435');
    expect(config.costPer1kTokens).toBe(0);
    expect(isEmbeddingConfigured(null)).toBe(true);
  });

  test('uses Ollama /api/embed and validates configured dimensions', async () => {
    process.env.GBRAIN_EMBEDDING_PROVIDER = 'ollama';
    process.env.GBRAIN_EMBEDDING_MODEL = 'test-embed';
    const vector = Array.from({ length: 1536 }, (_, i) => i / 1536);

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(JSON.parse(String(init?.body))).toEqual({
        model: 'test-embed',
        input: ['hello'],
      });
      return new Response(JSON.stringify({ embeddings: [vector] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }) as typeof fetch;

    try {
      const embeddings = await embedBatch(['hello']);
      expect(embeddings[0].length).toBe(1536);
      expect(embeddings[0][1]).toBeCloseTo(1 / 1536, 6);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
