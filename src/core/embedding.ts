/**
 * Embedding Service
 *
 * Defaults to OpenAI text-embedding-3-large at 1536 dimensions, matching the
 * fixed pgvector schema. OpenRouter uses the OpenAI-compatible embeddings API;
 * Ollama uses the local /api/embed endpoint and must return the configured
 * dimension count.
 */

import OpenAI from 'openai';
import { loadConfig, type GBrainConfig } from './config.ts';

const MODEL = 'text-embedding-3-large';
const OPENROUTER_MODEL = 'openai/text-embedding-3-large';
const OLLAMA_MODEL = 'rjmalagon/gte-qwen2-1.5b-instruct-embed-f16';
const DIMENSIONS = 1536;
const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;

export type EmbeddingProvider = 'openai' | 'openrouter' | 'ollama';

export interface ResolvedEmbeddingConfig {
  provider: EmbeddingProvider;
  model: string;
  dimensions: number;
  baseURL?: string;
  apiKey?: string;
  batchSize: number;
  costPer1kTokens: number;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated]);
  return result[0];
}

export interface EmbedBatchOptions {
  /**
   * Optional callback fired after each 100-item sub-batch completes.
   * CLI wrappers tick a reporter; Minion handlers can call
   * job.updateProgress here instead of hooking the per-page callback.
   */
  onBatchComplete?: (done: number, total: number) => void;
}

export async function embedBatch(
  texts: string[],
  options: EmbedBatchOptions = {},
): Promise<Float32Array[]> {
  const config = resolveEmbeddingConfig();
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  for (let i = 0; i < truncated.length; i += config.batchSize) {
    const batch = truncated.slice(i, i + config.batchSize);
    const batchResults = await embedBatchWithRetry(batch, config);
    results.push(...batchResults);
    options.onBatchComplete?.(results.length, truncated.length);
  }

  return results;
}

export function resolveEmbeddingConfig(config: GBrainConfig | null = loadConfig()): ResolvedEmbeddingConfig {
  const file = config?.embeddings ?? {};
  const envProvider = process.env.GBRAIN_EMBEDDING_PROVIDER as EmbeddingProvider | undefined;
  const provider = envProvider || file.provider || 'openai';
  if (!['openai', 'openrouter', 'ollama'].includes(provider)) {
    throw new Error(`Unsupported embedding provider: ${provider}`);
  }

  const dimensions = parsePositiveInt(process.env.GBRAIN_EMBEDDING_DIMENSIONS)
    ?? file.dimensions
    ?? DIMENSIONS;
  if (dimensions !== DIMENSIONS) {
    throw new Error(
      `Unsupported embedding dimensions: ${dimensions}. ` +
      `The current gbrain schema stores vector(${DIMENSIONS}); use a ${DIMENSIONS}-dimensional model.`,
    );
  }
  const model = process.env.GBRAIN_EMBEDDING_MODEL
    || file.model
    || defaultModel(provider);
  const baseURL = process.env.GBRAIN_EMBEDDING_BASE_URL
    || file.base_url
    || defaultBaseURL(provider);
  const apiKey = resolveApiKey(provider, config);
  const batchSize = parsePositiveInt(process.env.GBRAIN_EMBED_BATCH_SIZE) ?? BATCH_SIZE;
  const costPer1kTokens = parseFiniteNumber(process.env.GBRAIN_EMBEDDING_COST_PER_1K_TOKENS)
    ?? file.cost_per_1k_tokens
    ?? (provider === 'ollama' ? 0 : EMBEDDING_COST_PER_1K_TOKENS);

  return { provider, model, dimensions, baseURL, apiKey, batchSize, costPer1kTokens };
}

export function isEmbeddingConfigured(config: GBrainConfig | null = loadConfig()): boolean {
  const resolved = resolveEmbeddingConfig(config);
  if (resolved.provider === 'ollama') return true;
  return Boolean(resolved.apiKey);
}

export function getEmbeddingModel(): string {
  return resolveEmbeddingConfig().model;
}

async function embedBatchWithRetry(
  texts: string[],
  config: ResolvedEmbeddingConfig,
): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return config.provider === 'ollama'
        ? await embedWithOllama(texts, config)
        : await embedWithOpenAICompatible(texts, config);
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

const clients = new Map<string, OpenAI>();

function getOpenAICompatibleClient(config: ResolvedEmbeddingConfig): OpenAI {
  if (!config.apiKey) {
    const envVar = config.provider === 'openrouter' ? 'OPENROUTER_API_KEY' : 'OPENAI_API_KEY';
    throw new Error(`Missing embedding API key. Set ${envVar} or GBRAIN_EMBEDDING_API_KEY.`);
  }
  const key = `${config.provider}:${config.baseURL || ''}:${config.apiKey}`;
  let client = clients.get(key);
  if (!client) {
    client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    });
    clients.set(key, client);
  }
  return client;
}

async function embedWithOpenAICompatible(
  texts: string[],
  config: ResolvedEmbeddingConfig,
): Promise<Float32Array[]> {
  const response = await getOpenAICompatibleClient(config).embeddings.create({
    model: config.model,
    input: texts,
    dimensions: config.dimensions,
  });

  const sorted = response.data.sort((a, b) => a.index - b.index);
  return sorted.map(d => toCheckedFloat32Array(d.embedding, config));
}

async function embedWithOllama(
  texts: string[],
  config: ResolvedEmbeddingConfig,
): Promise<Float32Array[]> {
  const url = new URL('/api/embed', config.baseURL || defaultBaseURL('ollama')!);
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      input: texts,
    }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`Ollama embedding request failed (${response.status}): ${body || response.statusText}`);
  }

  const payload = await response.json() as { embeddings?: number[][]; embedding?: number[] };
  const embeddings = payload.embeddings ?? (payload.embedding ? [payload.embedding] : null);
  if (!embeddings || embeddings.length !== texts.length) {
    throw new Error(`Ollama returned ${embeddings?.length ?? 0} embeddings for ${texts.length} inputs`);
  }
  return embeddings.map(embedding => toCheckedFloat32Array(embedding, config));
}

function toCheckedFloat32Array(
  embedding: number[],
  config: ResolvedEmbeddingConfig,
): Float32Array {
  if (embedding.length !== config.dimensions) {
    throw new Error(
      `Embedding dimension mismatch for ${config.provider}/${config.model}: ` +
      `expected ${config.dimensions}, got ${embedding.length}. ` +
      `The current gbrain schema stores vector(${DIMENSIONS}); use a ${DIMENSIONS}-dimensional model.`,
    );
  }
  return new Float32Array(embedding);
}

function resolveApiKey(provider: EmbeddingProvider, config: GBrainConfig | null): string | undefined {
  if (process.env.GBRAIN_EMBEDDING_API_KEY) return process.env.GBRAIN_EMBEDDING_API_KEY;
  if (provider === 'openrouter') return process.env.OPENROUTER_API_KEY || config?.embeddings?.api_key;
  if (provider === 'openai') return process.env.OPENAI_API_KEY || config?.openai_api_key || config?.embeddings?.api_key;
  return undefined;
}

function defaultModel(provider: EmbeddingProvider): string {
  if (provider === 'openrouter') return OPENROUTER_MODEL;
  if (provider === 'ollama') return OLLAMA_MODEL;
  return MODEL;
}

function defaultBaseURL(provider: EmbeddingProvider): string | undefined {
  if (provider === 'openrouter') return 'https://openrouter.ai/api/v1';
  if (provider === 'ollama') return process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
  return undefined;
}

function parsePositiveInt(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function parseFiniteNumber(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export { MODEL as EMBEDDING_MODEL, DIMENSIONS as EMBEDDING_DIMENSIONS };

/**
 * v0.20.0 Cathedral II Layer 8 (D1): USD cost per 1k tokens for
 * text-embedding-3-large. Used by `gbrain sync --all` cost preview and
 * the reindex-code backfill command to surface expected spend before
 * the agent/user accepts an expensive operation.
 *
 * Value: $0.00013 / 1k tokens as of 2026. Update when OpenAI changes
 * pricing. Single source of truth — every cost-preview surface reads
 * this constant, so a pricing change is a one-line edit.
 */
export const EMBEDDING_COST_PER_1K_TOKENS = 0.00013;

/** Compute USD cost estimate for embedding `tokens` at current model rate. */
export function estimateEmbeddingCostUsd(tokens: number): number {
  return (tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS;
}
