export type InputFormat = "auto" | "csv" | "tsv" | "json" | "txt";
export type OutputFormat = "table" | "json" | "csv" | "markdown";
export type EmbeddingProvider = "local" | "openai";

export interface LoadOptions {
  format?: InputFormat;
  textColumn?: string;
  valueColumn?: string;
  frictionColumn?: string;
  delimiter?: string;
  dedupe?: boolean;
}

export interface PhraseRecord {
  phrase: string;
  sourceIndex: number;
  raw: Record<string, unknown>;
  value?: number;
  friction?: number;
}

export interface ScoredRecord extends PhraseRecord {
  score: number;
  normalizedValue: number;
  normalizedFriction: number;
  specificity: number;
  intent: string;
  wordCount: number;
  metricsSource: "columns" | "text-proxy" | "mixed";
}

export interface KeyphraseResult {
  phrase: string;
  score: number;
  count: number;
}

export interface ClusterAssignment {
  phrase: string;
  cluster: number;
  topic: string;
  size: number;
}

export interface ClusterResult {
  assignments: ClusterAssignment[];
  summary: Array<{
    cluster: number;
    topic: string;
    size: number;
    examples: string[];
  }>;
}

export interface GapResult {
  mineCount: number;
  competitorCount: number;
  sharedCount: number;
  uniqueToMine: string[];
  uniqueToCompetitor: string[];
  shared: string[];
}

export interface OpportunityModel {
  version: 2;
  modelType: "embedding-lightgbm";
  createdAt: string;
  embedding: {
    provider: EmbeddingProvider;
    model: string;
    dimensions: number;
  };
  lgbm: {
    objective: "regression";
    featureCount: number;
    initialScore: number;
    learningRate: number;
    trees: LightGbmTree[];
  };
  training: {
    samples: number;
    trees: number;
    loss: number;
    target: "opportunity-score";
  };
  notes: string;
}

export interface LightGbmTree {
  feature: number;
  threshold: number;
  leftValue: number;
  rightValue: number;
  gain: number;
}

export interface PredictionResult {
  phrase: string;
  predictedScore: number;
}

export interface BriefResult {
  keyword: string;
  related: string[];
  format: "markdown" | "json";
  content: string | Record<string, unknown>;
  generatedBy: "template" | "openai";
}

export interface NicheIdea {
  niche: string;
  score: number;
  reason: string;
  generatedBy: "template" | "openai";
}
