import type { PhraseRecord } from "./types.js";
import { isQuestionPhrase } from "./text.js";

export function findQuestionRecords(records: PhraseRecord[]): PhraseRecord[] {
  return records.filter((record) => isQuestionPhrase(record.phrase));
}
