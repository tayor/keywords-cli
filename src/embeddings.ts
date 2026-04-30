import type { EmbeddingProvider } from "./types.js";
import { buildNgrams, tokenize } from "./text.js";

export interface EmbeddingOptions {
  provider?: EmbeddingProvider;
  model?: string;
  apiKey?: string;
  dimensions?: number;
}

export interface EmbeddingResult {
  vectors: number[][];
  provider: EmbeddingProvider;
  model: string;
  dimensions: number;
}

interface OpenAiEmbeddingResponse {
  data?: Array<{ embedding?: number[]; index?: number }>;
  error?: { message?: string };
}

const DEFAULT_LOCAL_MODEL = "local-token-embedding-v1";
const DEFAULT_LOCAL_DIMENSIONS = 384;
const DEFAULT_OPENAI_MODEL = "text-embedding-3-small";

export async function embedPhrases(phrases: string[], options: EmbeddingOptions = {}): Promise<EmbeddingResult> {
  const provider = options.provider ?? "local";
  if (provider === "openai") {
    return embedWithOpenAi(phrases, options);
  }

  const dimensions = Math.max(16, options.dimensions ?? DEFAULT_LOCAL_DIMENSIONS);
  return {
    vectors: phrases.map((phrase) => makeLocalEmbedding(phrase, dimensions)),
    provider: "local",
    model: options.model ?? DEFAULT_LOCAL_MODEL,
    dimensions
  };
}

export function makeLocalEmbedding(phrase: string, dimensions = DEFAULT_LOCAL_DIMENSIONS): number[] {
  const safeDimensions = Math.max(16, dimensions);
  const tokens = tokenize(phrase, true);
  const grams = [...tokens, ...buildNgrams(tokens, 2, 3)];
  const vector = Array.from({ length: safeDimensions }, () => 0);

  for (const gram of grams) {
    const hash = stableHash(gram);
    const index = hash % safeDimensions;
    const sign = (hash & 1) === 0 ? 1 : -1;
    const weight = 1 + Math.min(2, gram.split(" ").length) * 0.15;
    vector[index] = (vector[index] ?? 0) + sign * weight;
  }

  return normalizeVector(vector);
}

async function embedWithOpenAi(phrases: string[], options: EmbeddingOptions): Promise<EmbeddingResult> {
  const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required for --embedding-provider openai.");
  }

  const model = options.model ?? DEFAULT_OPENAI_MODEL;
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({ model, input: phrases })
  });

  const payload = (await response.json()) as OpenAiEmbeddingResponse;
  if (!response.ok) {
    throw new Error(payload.error?.message ?? `OpenAI embeddings request failed with HTTP ${response.status}`);
  }

  const rows = [...(payload.data ?? [])].sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  const vectors = rows.map((row) => row.embedding ?? []);
  const dimensions = vectors[0]?.length ?? 0;
  if (vectors.length !== phrases.length || dimensions === 0 || vectors.some((vector) => vector.length !== dimensions)) {
    throw new Error("OpenAI embeddings response did not include one consistent vector per phrase.");
  }

  return {
    vectors: vectors.map(normalizeVector),
    provider: "openai",
    model,
    dimensions
  };
}

function normalizeVector(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (norm === 0) {
    return vector;
  }
  return vector.map((value) => value / norm);
}

function stableHash(input: string): number {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}