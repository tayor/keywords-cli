import type { KeyphraseResult } from "./types.js";
import { buildNgrams, jaccardSimilarity, normalizePhrase, round, tokenize } from "./text.js";

export interface ExtractOptions {
  topN?: number;
  minNgram?: number;
  maxNgram?: number;
  diversity?: number;
}

export function extractKeyphrases(documents: string[], options: ExtractOptions = {}): KeyphraseResult[] {
  const topN = Math.max(1, options.topN ?? 20);
  const minNgram = Math.max(1, options.minNgram ?? 1);
  const maxNgram = Math.max(minNgram, options.maxNgram ?? 3);
  const diversity = Math.max(0, Math.min(1, options.diversity ?? 0.65));

  const documentCandidates = documents.map((document) => {
    const tokens = tokenize(document, true);
    return buildNgrams(tokens, minNgram, maxNgram);
  });

  const termFrequency = new Map<string, number>();
  const documentFrequency = new Map<string, number>();

  for (const candidates of documentCandidates) {
    const seenInDoc = new Set<string>();
    for (const candidate of candidates) {
      if (!candidate || candidate.length < 2) {
        continue;
      }
      termFrequency.set(candidate, (termFrequency.get(candidate) ?? 0) + 1);
      seenInDoc.add(candidate);
    }
    for (const candidate of seenInDoc) {
      documentFrequency.set(candidate, (documentFrequency.get(candidate) ?? 0) + 1);
    }
  }

  const documentCount = Math.max(1, documents.length);
  const candidates = [...termFrequency.entries()].map(([phrase, count]) => {
    const df = documentFrequency.get(phrase) ?? 1;
    const idf = Math.log((documentCount + 1) / (df + 0.5)) + 1;
    const phraseLength = phrase.split(" ").length;
    const lengthBoost = 1 + Math.min(0.4, (phraseLength - 1) * 0.15);
    return {
      phrase,
      count,
      score: count * idf * lengthBoost
    };
  });

  const sorted = candidates.sort((left, right) => right.score - left.score);
  const selected: KeyphraseResult[] = [];
  const maxScore = sorted[0]?.score || 1;

  while (selected.length < topN && sorted.length > 0) {
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < sorted.length; index += 1) {
      const candidate = sorted[index];
      if (!candidate) {
        continue;
      }
      const relevance = candidate.score / maxScore;
      const similarity = selected.length === 0 ? 0 : Math.max(...selected.map((item) => jaccardSimilarity(candidate.phrase, item.phrase)));
      const mmrScore = relevance * (1 - diversity * 0.5) - similarity * diversity;
      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIndex = index;
      }
    }

    const [chosen] = sorted.splice(bestIndex, 1);
    if (!chosen || selected.some((item) => normalizePhrase(item.phrase) === normalizePhrase(chosen.phrase))) {
      continue;
    }
    selected.push({
      phrase: chosen.phrase,
      count: chosen.count,
      score: round(chosen.score / maxScore, 4)
    });
  }

  return selected;
}
