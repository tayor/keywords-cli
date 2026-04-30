import type { PhraseRecord, ScoredRecord } from "./types.js";
import { clamp, inferIntent, intentValueBonus, round, specificityScore, tokenize } from "./text.js";

export function scoreRecords(records: PhraseRecord[]): ScoredRecord[] {
  if (records.length === 0) {
    return [];
  }

  const valueSources = records.map((record) => record.value ?? estimateValueProxy(record.phrase));
  const frictionSources = records.map((record) => record.friction ?? estimateFrictionProxy(record.phrase));
  const normalizedValues = normalizeSeries(valueSources);
  const normalizedFrictions = normalizeSeries(frictionSources);
  const hasColumnValue = records.some((record) => typeof record.value === "number");
  const hasColumnFriction = records.some((record) => typeof record.friction === "number");

  return records
    .map((record, index) => {
      const intent = inferIntent(record.phrase);
      const specificity = specificityScore(record.phrase);
      const value = normalizedValues[index] ?? 0;
      const friction = normalizedFrictions[index] ?? 0;
      const columnCompleteness = record.value !== undefined && record.friction !== undefined;
      const score = computeOpportunityScore(value, friction, specificity, intent);

      return {
        ...record,
        score: round(score, 2),
        normalizedValue: round(value, 4),
        normalizedFriction: round(friction, 4),
        specificity: round(specificity, 4),
        intent,
        wordCount: tokenize(record.phrase).length,
        metricsSource: columnCompleteness ? "columns" : hasColumnValue || hasColumnFriction ? "mixed" : "text-proxy"
      } satisfies ScoredRecord;
    })
    .sort((left, right) => right.score - left.score);
}

export function computeOpportunityScore(
  normalizedValue: number,
  normalizedFriction: number,
  specificity: number,
  intent: string
): number {
  const composite = normalizedValue / (normalizedFriction + 0.05);
  const compressedComposite = composite / (composite + 1);
  const boosted = compressedComposite * 0.82 + specificity * 0.12 + intentValueBonus(intent) * 0.06;
  return clamp(boosted * 100, 0, 100);
}

export function estimateValueProxy(phrase: string): number {
  const intent = inferIntent(phrase);
  const specificity = specificityScore(phrase);
  const wordCount = tokenize(phrase).length;
  const longTailDemand = Math.max(0, 1 - Math.abs(wordCount - 4) / 5);
  return 20 + longTailDemand * 45 + specificity * 20 + intentValueBonus(intent) * 40;
}

export function estimateFrictionProxy(phrase: string): number {
  const tokens = tokenize(phrase, true);
  const wordCount = tokenize(phrase).length;
  const broadness = Math.max(0, 1 - Math.min(1, (wordCount - 1) / 5));
  const genericPenalty = tokens.length <= 1 ? 35 : 0;
  const headTermPenalty = wordCount <= 2 ? 20 : 0;
  return 25 + broadness * 35 + genericPenalty + headTermPenalty;
}

function normalizeSeries(values: number[]): number[] {
  const finiteValues = values.filter(Number.isFinite);
  if (finiteValues.length === 0) {
    return values.map(() => 0);
  }
  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);
  const range = max - min;
  if (range === 0) {
    return values.map(() => 0.5);
  }
  return values.map((value) => (Number.isFinite(value) ? clamp((value - min) / range, 0, 1) : 0));
}
