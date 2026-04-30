import type { GapResult, PhraseRecord } from "./types.js";
import { normalizePhrase } from "./text.js";

export function compareGaps(mine: PhraseRecord[], competitor: PhraseRecord[]): GapResult {
  const mineMap = toNormalizedMap(mine);
  const competitorMap = toNormalizedMap(competitor);
  const mineKeys = new Set(mineMap.keys());
  const competitorKeys = new Set(competitorMap.keys());

  const uniqueToMine = [...mineKeys]
    .filter((key) => !competitorKeys.has(key))
    .map((key) => mineMap.get(key) ?? key)
    .sort();
  const uniqueToCompetitor = [...competitorKeys]
    .filter((key) => !mineKeys.has(key))
    .map((key) => competitorMap.get(key) ?? key)
    .sort();
  const shared = [...mineKeys]
    .filter((key) => competitorKeys.has(key))
    .map((key) => mineMap.get(key) ?? key)
    .sort();

  return {
    mineCount: mineKeys.size,
    competitorCount: competitorKeys.size,
    sharedCount: shared.length,
    uniqueToMine,
    uniqueToCompetitor,
    shared
  };
}

function toNormalizedMap(records: PhraseRecord[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const record of records) {
    const key = normalizePhrase(record.phrase);
    if (!map.has(key)) {
      map.set(key, record.phrase);
    }
  }
  return map;
}
