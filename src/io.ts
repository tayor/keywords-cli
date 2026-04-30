import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import Papa from "papaparse";
import type { InputFormat, LoadOptions, PhraseRecord } from "./types.js";
import { normalizePhrase } from "./text.js";

const TEXT_COLUMN_HINTS = [
  "keyword",
  "keywords",
  "keyword phrase",
  "query",
  "term",
  "phrase",
  "topic",
  "niche",
  "idea",
  "title",
  "text",
  "name"
];

const VALUE_COLUMN_HINTS = [
  "search volume",
  "volume",
  "demand",
  "traffic",
  "revenue",
  "value",
  "sales",
  "conversions",
  "priority",
  "impact",
  "potential"
];

const FRICTION_COLUMN_HINTS = [
  "ranking difficulty",
  "difficulty",
  "competition",
  "cost",
  "effort",
  "complexity",
  "risk",
  "friction",
  "saturation"
];

export async function loadRecords(input: string, options: LoadOptions = {}): Promise<PhraseRecord[]> {
  const content = await readInput(input);
  const format = resolveFormat(input, options.format ?? "auto", content);

  let records: PhraseRecord[];
  if (format === "json") {
    records = recordsFromJson(content, options);
  } else if (format === "csv" || format === "tsv") {
    records = recordsFromDelimited(content, { ...options, delimiter: options.delimiter ?? (format === "tsv" ? "\t" : undefined) });
  } else {
    records = recordsFromText(content);
  }

  return options.dedupe === false ? records : dedupeRecords(records);
}

export async function readInput(input: string): Promise<string> {
  if (input === "-") {
    return readStdin();
  }
  if (existsSync(input)) {
    return readFile(input, "utf8");
  }
  return input;
}

export function recordsFromText(content: string): PhraseRecord[] {
  return content
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((phrase, sourceIndex) => ({
      phrase,
      sourceIndex,
      raw: { phrase }
    }));
}

export function recordsFromJson(content: string, options: LoadOptions = {}): PhraseRecord[] {
  const parsed = JSON.parse(content) as unknown;
  const items = unwrapJsonItems(parsed);

  if (!Array.isArray(items)) {
    throw new Error("JSON input must be an array or an object with a keywords/items/data/records array.");
  }

  return items
    .map((item, sourceIndex) => recordFromUnknown(item, sourceIndex, options))
    .filter((record): record is PhraseRecord => Boolean(record));
}

export function recordsFromDelimited(content: string, options: LoadOptions = {}): PhraseRecord[] {
  const delimiter = options.delimiter;
  const firstLine = content.split(/\r?\n/u).find((line) => line.trim()) ?? "";
  const headerParse = Papa.parse<Record<string, string>>(content, {
    header: true,
    skipEmptyLines: true,
    delimiter
  });

  if (headerParse.errors.length > 0) {
    throw new Error(`Unable to parse delimited input: ${headerParse.errors[0]?.message ?? "unknown parse error"}`);
  }

  const fields = headerParse.meta.fields?.filter(Boolean) ?? [];
  const hasRequestedColumns = Boolean(options.textColumn || options.valueColumn || options.frictionColumn);
  const shouldUseHeader = fields.length > 0 && (hasRequestedColumns || fields.some(looksLikeHeader));

  if (shouldUseHeader && fields.length > 0) {
    const textColumn = pickColumn(fields, options.textColumn, TEXT_COLUMN_HINTS);
    if (textColumn) {
      const valueColumn = pickColumn(fields, options.valueColumn, VALUE_COLUMN_HINTS);
      const frictionColumn = pickColumn(fields, options.frictionColumn, FRICTION_COLUMN_HINTS);
      return headerParse.data
        .map((row, sourceIndex) => recordFromObject(row, sourceIndex, textColumn, valueColumn, frictionColumn))
        .filter((record): record is PhraseRecord => Boolean(record));
    }
  }

  const rowParse = Papa.parse<string[]>(content, {
    header: false,
    skipEmptyLines: true,
    delimiter
  });
  if (rowParse.errors.length > 0) {
    throw new Error(`Unable to parse delimited input: ${rowParse.errors[0]?.message ?? "unknown parse error"}`);
  }

  return rowParse.data
    .map((row, sourceIndex): PhraseRecord | undefined => {
      const phrase = String(row[0] ?? "").trim();
      if (!phrase) {
        return undefined;
      }
      return {
        phrase,
        sourceIndex,
        raw: Object.fromEntries(row.map((value, index) => [`column_${index + 1}`, value]))
      };
    })
    .filter((record): record is PhraseRecord => Boolean(record));
}

export function pickColumn(columns: string[], requested: string | undefined, hints: string[]): string | undefined {
  if (requested) {
    const exact = columns.find((column) => normalizeHeader(column) === normalizeHeader(requested));
    if (!exact) {
      throw new Error(`Column "${requested}" was not found. Available columns: ${columns.join(", ")}`);
    }
    return exact;
  }

  for (const hint of hints) {
    const exact = columns.find((column) => normalizeHeader(column) === normalizeHeader(hint));
    if (exact) {
      return exact;
    }
  }

  for (const hint of hints) {
    const partial = columns.find((column) => normalizeHeader(column).includes(normalizeHeader(hint)));
    if (partial) {
      return partial;
    }
  }

  return undefined;
}

function recordFromUnknown(item: unknown, sourceIndex: number, options: LoadOptions): PhraseRecord | undefined {
  if (typeof item === "string" || typeof item === "number") {
    const phrase = String(item).trim();
    return phrase ? { phrase, sourceIndex, raw: { phrase } } : undefined;
  }

  if (!item || typeof item !== "object" || Array.isArray(item)) {
    return undefined;
  }

  const object = item as Record<string, unknown>;
  const columns = Object.keys(object);
  const textColumn = pickColumn(columns, options.textColumn, TEXT_COLUMN_HINTS) ?? columns[0];
  const valueColumn = pickColumn(columns, options.valueColumn, VALUE_COLUMN_HINTS);
  const frictionColumn = pickColumn(columns, options.frictionColumn, FRICTION_COLUMN_HINTS);
  return recordFromObject(object, sourceIndex, textColumn, valueColumn, frictionColumn);
}

function recordFromObject(
  row: Record<string, unknown>,
  sourceIndex: number,
  textColumn: string,
  valueColumn?: string,
  frictionColumn?: string
): PhraseRecord | undefined {
  const phrase = String(row[textColumn] ?? "").trim();
  if (!phrase) {
    return undefined;
  }

  const value = parseNumeric(row[valueColumn ?? ""]);
  const friction = parseNumeric(row[frictionColumn ?? ""]);

  return {
    phrase,
    sourceIndex,
    raw: row,
    ...(value === undefined ? {} : { value }),
    ...(friction === undefined ? {} : { friction })
  };
}

function parseNumeric(value: unknown): number | undefined {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  const cleaned = String(value).replace(/,/gu, "").replace(/%/gu, "").trim();
  if (!cleaned) {
    return undefined;
  }
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function dedupeRecords(records: PhraseRecord[]): PhraseRecord[] {
  const seen = new Set<string>();
  const output: PhraseRecord[] = [];

  for (const record of records) {
    const key = normalizePhrase(record.phrase);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    output.push(record);
  }

  return output;
}

function resolveFormat(input: string, requested: InputFormat, content: string): Exclude<InputFormat, "auto"> {
  if (requested !== "auto") {
    return requested;
  }

  if (existsSync(input)) {
    const extension = extname(input).toLowerCase();
    if (extension === ".csv") {
      return "csv";
    }
    if (extension === ".tsv") {
      return "tsv";
    }
    if (extension === ".json") {
      return "json";
    }
    return "txt";
  }

  const trimmed = content.trim();
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    return "json";
  }
  if (trimmed.includes(",") && trimmed.includes("\n")) {
    return "csv";
  }
  return "txt";
}

function unwrapJsonItems(parsed: unknown): unknown {
  if (Array.isArray(parsed)) {
    return parsed;
  }
  if (!parsed || typeof parsed !== "object") {
    return parsed;
  }
  const object = parsed as Record<string, unknown>;
  for (const key of ["keywords", "items", "data", "records", "phrases", "ideas"]) {
    if (Array.isArray(object[key])) {
      return object[key];
    }
  }
  return parsed;
}

function normalizeHeader(header: string): string {
  return normalizePhrase(header).replace(/[_-]+/gu, " ");
}

function looksLikeHeader(value: string): boolean {
  const normalized = normalizeHeader(value);
  return [...TEXT_COLUMN_HINTS, ...VALUE_COLUMN_HINTS, ...FRICTION_COLUMN_HINTS].some(
    (hint) => normalized === normalizeHeader(hint)
  );
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  }
  return Buffer.concat(chunks).toString("utf8");
}
