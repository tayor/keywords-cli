const DEFAULT_STOPWORDS = new Set([
  "a",
  "about",
  "after",
  "all",
  "also",
  "an",
  "and",
  "any",
  "are",
  "as",
  "at",
  "be",
  "best",
  "but",
  "by",
  "can",
  "for",
  "from",
  "get",
  "has",
  "have",
  "how",
  "in",
  "into",
  "is",
  "it",
  "its",
  "near",
  "new",
  "of",
  "on",
  "or",
  "our",
  "that",
  "the",
  "their",
  "this",
  "to",
  "top",
  "up",
  "use",
  "vs",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "with",
  "without",
  "you",
  "your"
]);

const QUESTION_WORDS = new Set([
  "who",
  "what",
  "where",
  "when",
  "why",
  "how",
  "which",
  "can",
  "does",
  "do",
  "did",
  "is",
  "are",
  "will",
  "should"
]);

const COMMERCIAL_WORDS = new Set([
  "best",
  "review",
  "reviews",
  "alternative",
  "alternatives",
  "compare",
  "comparison",
  "vs",
  "pricing",
  "cheap",
  "affordable",
  "premium",
  "buy",
  "deal",
  "software",
  "tool",
  "template",
  "service"
]);

const TRANSACTIONAL_WORDS = new Set([
  "buy",
  "order",
  "download",
  "hire",
  "book",
  "coupon",
  "trial",
  "subscribe",
  "shop",
  "near"
]);

const LOCAL_WORDS = new Set(["near", "local", "nearby", "city", "map", "open", "hours"]);

export function normalizePhrase(input: string): string {
  return input
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, "\"")
    .replace(/\s+/g, " ")
    .trim();
}

export function tokenize(input: string, removeStopwords = false): string[] {
  const normalized = normalizePhrase(input);
  const matches = normalized.match(/[\p{L}\p{N}][\p{L}\p{N}'_-]*/gu) ?? [];
  const tokens = matches.map(simpleStem).filter(Boolean);
  if (!removeStopwords) {
    return tokens;
  }
  return tokens.filter((token) => !DEFAULT_STOPWORDS.has(token));
}

export function buildNgrams(tokens: string[], minNgram = 1, maxNgram = 3): string[] {
  const phrases: string[] = [];
  const min = Math.max(1, minNgram);
  const max = Math.max(min, maxNgram);

  for (let size = min; size <= max; size += 1) {
    for (let index = 0; index <= tokens.length - size; index += 1) {
      const slice = tokens.slice(index, index + size);
      if (slice.every((token) => DEFAULT_STOPWORDS.has(token))) {
        continue;
      }
      phrases.push(slice.join(" "));
    }
  }

  return phrases;
}

export function isQuestionPhrase(phrase: string): boolean {
  const normalized = normalizePhrase(phrase);
  const first = normalized.split(" ")[0] ?? "";
  return QUESTION_WORDS.has(first) || normalized.includes("?");
}

export function inferIntent(phrase: string): string {
  const normalized = normalizePhrase(phrase);
  const tokens = new Set(tokenize(normalized));

  if (isQuestionPhrase(phrase)) {
    return "question";
  }
  if ([...tokens].some((token) => TRANSACTIONAL_WORDS.has(token))) {
    return "transactional";
  }
  if ([...tokens].some((token) => LOCAL_WORDS.has(token))) {
    return "local";
  }
  if (tokens.has("vs") || tokens.has("versus") || tokens.has("compare")) {
    return "comparison";
  }
  if ([...tokens].some((token) => COMMERCIAL_WORDS.has(token))) {
    return "commercial";
  }
  if (tokens.has("problem") || tokens.has("fix") || tokens.has("solve")) {
    return "problem";
  }
  if (tokens.has("guide") || tokens.has("tutorial") || tokens.has("learn")) {
    return "informational";
  }
  return "generic";
}

export function specificityScore(phrase: string): number {
  const tokens = tokenize(phrase, true);
  const wordCount = tokenize(phrase).length;
  const uniqueRatio = tokens.length === 0 ? 0 : new Set(tokens).size / tokens.length;
  const lengthScore = Math.min(1, Math.max(0, (wordCount - 1) / 5));
  const modifierScore = hasLongTailModifier(phrase) ? 0.2 : 0;
  return clamp(lengthScore * 0.65 + uniqueRatio * 0.25 + modifierScore, 0, 1);
}

export function intentValueBonus(intent: string): number {
  switch (intent) {
    case "transactional":
      return 0.25;
    case "commercial":
    case "comparison":
      return 0.18;
    case "question":
    case "problem":
      return 0.12;
    case "local":
      return 0.1;
    case "informational":
      return 0.07;
    default:
      return 0;
  }
}

export function cosineSimilarity(left: Map<string, number>, right: Map<string, number>): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (const value of left.values()) {
    leftNorm += value * value;
  }
  for (const value of right.values()) {
    rightNorm += value * value;
  }

  const [small, large] = left.size <= right.size ? [left, right] : [right, left];
  for (const [key, value] of small) {
    dot += value * (large.get(key) ?? 0);
  }

  if (leftNorm === 0 || rightNorm === 0) {
    return 0;
  }
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export function jaccardSimilarity(leftPhrase: string, rightPhrase: string): number {
  const left = new Set(tokenize(leftPhrase, true));
  const right = new Set(tokenize(rightPhrase, true));
  if (left.size === 0 && right.size === 0) {
    return 0;
  }
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) {
      intersection += 1;
    }
  }
  const union = new Set([...left, ...right]).size;
  return union === 0 ? 0 : intersection / union;
}

export function makeTermVector(text: string): Map<string, number> {
  const tokens = tokenize(text, true);
  const features = [...tokens, ...buildNgrams(tokens, 2, 2)];
  const vector = new Map<string, number>();
  for (const feature of features) {
    vector.set(feature, (vector.get(feature) ?? 0) + 1);
  }
  return vector;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function round(value: number, digits = 2): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

export function stopwords(): Set<string> {
  return new Set(DEFAULT_STOPWORDS);
}

function simpleStem(token: string): string {
  if (token.length <= 4) {
    return token;
  }
  return token
    .replace(/'(s|re|ve|ll|d)$/u, "")
    .replace(/(ing|edly|edly|ed|ly)$/u, "")
    .replace(/(ies)$/u, "y")
    .replace(/(s)$/u, "");
}

function hasLongTailModifier(phrase: string): boolean {
  const tokens = new Set(tokenize(phrase));
  return (
    tokens.has("for") ||
    tokens.has("with") ||
    tokens.has("without") ||
    tokens.has("near") ||
    tokens.has("alternative") ||
    tokens.has("template")
  );
}
