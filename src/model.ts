import type { EmbeddingProvider, OpportunityModel, PhraseRecord, PredictionResult } from "./types.js";
import { embedPhrases } from "./embeddings.js";
import { predictLightGbmRegressor, trainLightGbmRegressor } from "./lgbm.js";
import { scoreRecords } from "./scoring.js";
import { clamp } from "./text.js";

export interface TrainOptions {
  embeddingProvider?: EmbeddingProvider;
  embeddingModel?: string;
  apiKey?: string;
  dimensions?: number;
  hashSize?: number;
  trees?: number;
  learningRate?: number;
}

export interface PredictOptions {
  apiKey?: string;
  embeddingProvider?: EmbeddingProvider;
  embeddingModel?: string;
}

export async function trainOpportunityModel(records: PhraseRecord[], options: TrainOptions = {}): Promise<OpportunityModel> {
  if (records.length < 2) {
    throw new Error("Training requires at least two phrases with usable value/friction signals.");
  }

  const scored = scoreRecords(records);
  const targets = scored.map((record) => record.score / 100);
  const embeddings = await embedPhrases(scored.map((record) => record.phrase), {
    provider: options.embeddingProvider ?? "local",
    model: options.embeddingModel,
    apiKey: options.apiKey,
    dimensions: options.dimensions ?? options.hashSize
  });
  const trained = trainLightGbmRegressor(embeddings.vectors, targets, {
    trees: options.trees,
    learningRate: options.learningRate
  });

  return {
    version: 2,
    modelType: "embedding-lightgbm",
    createdAt: new Date().toISOString(),
    embedding: {
      provider: embeddings.provider,
      model: embeddings.model,
      dimensions: embeddings.dimensions
    },
    lgbm: trained.model,
    training: {
      samples: records.length,
      trees: trained.model.trees.length,
      loss: trained.loss,
      target: "opportunity-score"
    },
    notes: "Embedding-backed LGBM-style boosted tree model trained from generic value/friction opportunity scores. Use --embedding-provider openai for direct OpenAI embeddings; local embeddings are deterministic for offline tests."
  };
}

export async function predictWithModel(
  records: PhraseRecord[],
  model: OpportunityModel,
  options: PredictOptions = {}
): Promise<PredictionResult[]> {
  const embeddings = await embedPhrases(records.map((record) => record.phrase), {
    provider: options.embeddingProvider ?? model.embedding.provider,
    model: options.embeddingModel ?? model.embedding.model,
    apiKey: options.apiKey,
    dimensions: model.embedding.dimensions
  });
  const predictions = predictLightGbmRegressor(embeddings.vectors, model.lgbm);
  return records
    .map((record, index) => ({
      phrase: record.phrase,
      predictedScore: Math.round(clamp((predictions[index] ?? 0) * 100, 0, 100) * 100) / 100
    }))
    .sort((left, right) => right.predictedScore - left.predictedScore);
}
