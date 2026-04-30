#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { Command } from "commander";
import { generateBrief, discoverNiches } from "./ai.js";
import { clusterRecords } from "./clustering.js";
import { extractKeyphrases } from "./extraction.js";
import { compareGaps } from "./gaps.js";
import { loadRecords } from "./io.js";
import { trainOpportunityModel, predictWithModel } from "./model.js";
import { emitOutput, normalizeFormat } from "./output.js";
import { findQuestionRecords } from "./questions.js";
import { scoreRecords } from "./scoring.js";
import type { EmbeddingProvider, LoadOptions, OpportunityModel, OutputFormat } from "./types.js";

const VERSION = "0.1.0";

const program = new Command();

program
  .name("keywords-cli")
  .description("Phrase research CLI for keywords, niches, product ideas, support topics, and any list with value/friction signals.")
  .version(VERSION);

program
  .command("analyze")
  .description("Score phrases by opportunity using value/friction columns or text-only proxy signals.")
  .argument("<input>", "CSV/TSV/JSON/TXT path, '-' for stdin, or raw newline-separated text")
  .option("--text-column <name>", "phrase column name")
  .option("--value-column <name>", "positive metric column, such as Search Volume, demand, revenue, or impact")
  .option("--friction-column <name>", "negative metric column, such as Ranking Difficulty, competition, cost, or effort")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--top <number>", "number of ranked rows to return", parsePositiveInt, 25)
  .option("--out <file>", "write output to a file")
  .action(async (input: string, options: CommonInputOptions & TopOptions & OutputOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const scored = scoreRecords(records).slice(0, options.top);
      const rows = scored.map((record) => ({
        phrase: record.phrase,
        score: record.score,
        intent: record.intent,
        words: record.wordCount,
        value: record.value ?? "",
        friction: record.friction ?? "",
        specificity: record.specificity,
        source: record.metricsSource
      }));
      await print(rows, options);
    });
  });

program
  .command("extract")
  .description("Extract KeyBERT-style keyphrases with local n-gram scoring and diversity filtering.")
  .argument("<input>", "file path, '-' for stdin, or raw text")
  .option("--text-column <name>", "phrase/text column name for structured files")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--top <number>", "number of keyphrases", parsePositiveInt, 20)
  .option("--min-ngram <number>", "minimum n-gram length", parsePositiveInt, 1)
  .option("--max-ngram <number>", "maximum n-gram length", parsePositiveInt, 3)
  .option("--diversity <number>", "0-1 diversity penalty", parseNumber, 0.65)
  .option("--out <file>", "write output to a file")
  .action(async (input: string, options: CommonInputOptions & TopOptions & OutputOptions & ExtractOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const keyphrases = extractKeyphrases(records.map((record) => record.phrase), {
        topN: options.top,
        minNgram: options.minNgram,
        maxNgram: options.maxNgram,
        diversity: options.diversity
      });
      await print(keyphrases, options);
    });
  });

program
  .command("cluster")
  .description("Cluster related phrases with a local similarity graph and label each cluster.")
  .argument("<input>", "CSV/TSV/JSON/TXT path, '-' for stdin, or raw newline-separated text")
  .option("--text-column <name>", "phrase column name")
  .option("--min-similarity <number>", "cosine similarity threshold", parseNumber, 0.32)
  .option("--min-size <number>", "minimum cluster size", parsePositiveInt, 2)
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--out <file>", "write output to a file")
  .action(async (input: string, options: CommonInputOptions & OutputOptions & ClusterOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const clusters = clusterRecords(records, {
        minSimilarity: options.minSimilarity,
        minClusterSize: options.minSize
      });
      const format = normalizeFormat(options.format);
      await print(format === "json" ? clusters : clusters.assignments, options);
    });
  });

program
  .command("questions")
  .description("Find question-style phrases for FAQ, support, research, and content workflows.")
  .argument("<input>", "CSV/TSV/JSON/TXT path, '-' for stdin, or raw newline-separated text")
  .option("--text-column <name>", "phrase column name")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--out <file>", "write output to a file")
  .action(async (input: string, options: CommonInputOptions & OutputOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const questions = findQuestionRecords(records).map((record) => ({ phrase: record.phrase }));
      await print(questions, options);
    });
  });

program
  .command("gap")
  .description("Compare your phrase list against another list and surface missing opportunities.")
  .requiredOption("--mine <file>", "your CSV/JSON/TXT phrase list")
  .requiredOption("--competitor <file>", "competitor or comparison phrase list")
  .option("--text-column <name>", "phrase column name")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--out <file>", "write output to a file")
  .action(async (options: GapOptions & CommonInputOptions & OutputOptions) => {
    await run(async () => {
      const mine = await loadRecords(options.mine, toLoadOptions(options));
      const competitor = await loadRecords(options.competitor, toLoadOptions(options));
      const result = compareGaps(mine, competitor);
      const format = normalizeFormat(options.format);
      if (format === "json") {
        await print(result, options);
        return;
      }
      const rows = [
        ...result.uniqueToCompetitor.map((phrase) => ({ bucket: "missing", phrase })),
        ...result.uniqueToMine.map((phrase) => ({ bucket: "mine_only", phrase })),
        ...result.shared.map((phrase) => ({ bucket: "shared", phrase }))
      ];
      await print(rows, options);
    });
  });

program
  .command("train")
  .description("Train an embedding + LGBM-style opportunity model from phrase value/friction data.")
  .argument("<input>", "CSV/TSV/JSON file with phrase and optional value/friction columns")
  .option("--text-column <name>", "phrase column name")
  .option("--value-column <name>", "positive metric column")
  .option("--friction-column <name>", "negative metric column")
  .option("--model <file>", "model output path", "opportunity-model.json")
  .option("--embedding-provider <provider>", "local or openai", "local")
  .option("--embedding-model <name>", "embedding model name; defaults to local-token-embedding-v1 or text-embedding-3-small")
  .option("--api-key <key>", "OpenAI API key for --embedding-provider openai; defaults to OPENAI_API_KEY")
  .option("--trees <number>", "number of boosted trees", parsePositiveInt, 80)
  .option("--iterations <number>", "alias for --trees", parsePositiveInt)
  .option("--learning-rate <number>", "boosted-tree learning rate", parseNumber, 0.08)
  .option("--dimensions <number>", "local embedding dimensions", parsePositiveInt, 384)
  .option("--hash-size <number>", "alias for --dimensions", parsePositiveInt)
  .action(async (input: string, options: TrainCliOptions & CommonInputOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const model = await trainOpportunityModel(records, {
        embeddingProvider: parseEmbeddingProvider(options.embeddingProvider),
        embeddingModel: options.embeddingModel,
        apiKey: options.apiKey,
        trees: options.iterations ?? options.trees,
        learningRate: options.learningRate,
        dimensions: options.hashSize ?? options.dimensions
      });
      await writeFile(options.model, `${JSON.stringify(model, null, 2)}\n`, "utf8");
      process.stdout.write(`Saved model to ${options.model}\nSamples: ${model.training.samples}\nTrees: ${model.training.trees}\nLoss: ${model.training.loss.toFixed(6)}\n`);
    });
  });

program
  .command("predict")
  .description("Predict opportunity scores for new phrases using a model created by train.")
  .argument("<input>", "CSV/TSV/JSON/TXT path, '-' for stdin, or raw newline-separated text")
  .requiredOption("--model <file>", "model JSON created by train")
  .option("--text-column <name>", "phrase column name")
  .option("--api-key <key>", "OpenAI API key when the model uses OpenAI embeddings; defaults to OPENAI_API_KEY")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--top <number>", "number of ranked rows to return", parsePositiveInt, 25)
  .option("--out <file>", "write output to a file")
  .action(async (input: string, options: PredictOptions & CommonInputOptions & TopOptions & OutputOptions) => {
    await run(async () => {
      const records = await loadRecords(input, toLoadOptions(options));
      const model = JSON.parse(await readFile(options.model, "utf8")) as OpportunityModel;
      const predictions = (await predictWithModel(records, model, { apiKey: options.apiKey })).slice(0, options.top);
      await print(predictions, options);
    });
  });

program
  .command("brief")
  .description("Generate a content/research brief. Uses OpenAI only when --ai is set and OPENAI_API_KEY exists.")
  .argument("<keyword>", "primary phrase or topic")
  .option("--related <items>", "comma-separated related phrases")
  .option("--related-file <file>", "file containing related phrases")
  .option("--ai", "use OpenAI instead of the deterministic template")
  .option("--model <name>", "OpenAI model", "gpt-4o-mini")
  .option("--api-key <key>", "OpenAI API key; defaults to OPENAI_API_KEY")
  .option("--format <format>", "markdown or json", "markdown")
  .option("--out <file>", "write output to a file")
  .action(async (keyword: string, options: BriefOptions & OutputOptions) => {
    await run(async () => {
      const related = await resolveRelated(options);
      const brief = await generateBrief(keyword, related, {
        useAi: options.ai,
        model: options.model,
        apiKey: options.apiKey
      });
      const requested = normalizeFormat(options.format === "table" ? "markdown" : options.format);
      if (requested === "json") {
        await print(brief, { ...options, format: "json" });
        return;
      }
      const markdown = brief.generatedBy === "openai" && typeof brief.content === "string"
        ? brief.content
        : briefToMarkdown(brief.content as Record<string, unknown>, keyword);
      await writeOrStdout(markdown.endsWith("\n") ? markdown : `${markdown}\n`, options.out);
    });
  });

program
  .command("discover")
  .description("Generate and rank niche ideas for a broad domain. Uses OpenAI only with --ai and an API key.")
  .argument("<domain>", "broad domain to explore")
  .option("--count <number>", "number of ideas", parsePositiveInt, 10)
  .option("--ai", "use OpenAI for generation")
  .option("--model <name>", "OpenAI model", "gpt-4o-mini")
  .option("--api-key <key>", "OpenAI API key; defaults to OPENAI_API_KEY")
  .option("--format <format>", "table, json, csv, or markdown", "table")
  .option("--out <file>", "write output to a file")
  .action(async (domain: string, options: DiscoverOptions & OutputOptions) => {
    await run(async () => {
      const ideas = await discoverNiches(domain, options.count, {
        useAi: options.ai,
        model: options.model,
        apiKey: options.apiKey
      });
      await print(ideas, options);
    });
  });

program.addHelpText(
  "after",
  `
Examples:
  keywords-cli analyze keywords.csv --value-column "Search Volume" --friction-column "Ranking Difficulty"
  keywords-cli analyze ideas.csv --text-column idea --value-column impact --friction-column effort
  keywords-cli extract notes.txt --top 30
  keywords-cli cluster keywords.txt --min-similarity 0.38
  keywords-cli train keywords.csv --model model.json --embedding-provider openai
  keywords-cli predict new-keywords.txt --model model.json
  keywords-cli brief "ai coloring book prompts" --related "midjourney prompts,coloring pages"
  keywords-cli discover "sustainable living products" --ai
`
);

program.parseAsync(process.argv).catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});

interface CommonInputOptions {
  textColumn?: string;
  valueColumn?: string;
  frictionColumn?: string;
}

interface OutputOptions {
  format?: string;
  out?: string;
}

interface TopOptions {
  top: number;
}

interface ExtractOptions {
  minNgram: number;
  maxNgram: number;
  diversity: number;
}

interface ClusterOptions {
  minSimilarity: number;
  minSize: number;
}

interface GapOptions {
  mine: string;
  competitor: string;
}

interface TrainCliOptions {
  model: string;
  embeddingProvider: string;
  embeddingModel?: string;
  apiKey?: string;
  trees: number;
  iterations?: number;
  learningRate: number;
  dimensions: number;
  hashSize?: number;
}

interface PredictOptions {
  model: string;
  apiKey?: string;
}

interface BriefOptions {
  related?: string;
  relatedFile?: string;
  ai?: boolean;
  model?: string;
  apiKey?: string;
}

interface DiscoverOptions {
  count: number;
  ai?: boolean;
  model?: string;
  apiKey?: string;
}

function toLoadOptions(options: CommonInputOptions): LoadOptions {
  return {
    textColumn: options.textColumn,
    valueColumn: options.valueColumn,
    frictionColumn: options.frictionColumn
  };
}

async function print(data: unknown, options: OutputOptions): Promise<void> {
  const format = normalizeFormat(options.format);
  const rendered = await emitOutput(data, format, options.out);
  if (!options.out) {
    process.stdout.write(rendered);
  } else {
    process.stderr.write(`Wrote ${options.out}\n`);
  }
}

async function writeOrStdout(content: string, out?: string): Promise<void> {
  if (out) {
    await writeFile(out, content, "utf8");
    process.stderr.write(`Wrote ${out}\n`);
    return;
  }
  process.stdout.write(content);
}

async function resolveRelated(options: BriefOptions): Promise<string[]> {
  const related = new Set<string>();
  if (options.related) {
    for (const item of options.related.split(",")) {
      const phrase = item.trim();
      if (phrase) {
        related.add(phrase);
      }
    }
  }
  if (options.relatedFile) {
    const records = await loadRecords(options.relatedFile);
    for (const record of records) {
      related.add(record.phrase);
    }
  }
  return [...related];
}

function briefToMarkdown(content: Record<string, unknown>, keyword: string): string {
  const titleIdeas = asStringArray(content.titleIdeas);
  const outline = asStringArray(content.outline);
  const questions = asStringArray(content.questions);
  const related = asStringArray(content.relatedKeywords);
  return [
    `# ${keyword}`,
    "",
    `Audience intent: ${String(content.audienceIntent ?? "topic exploration")}`,
    `Meta description: ${String(content.metaDescription ?? "")}`,
    `Word count target: ${String(content.wordCountTarget ?? "")}`,
    "",
    "## Title Ideas",
    ...titleIdeas.map((item) => `- ${item}`),
    "",
    "## Outline",
    ...outline.map((item) => `- ${item}`),
    "",
    "## Questions",
    ...questions.map((item) => `- ${item}`),
    "",
    "## Related",
    ...(related.length > 0 ? related.map((item) => `- ${item}`) : ["- None provided"])
  ].join("\n");
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item));
}

async function run(fn: () => Promise<void>): Promise<void> {
  try {
    await fn();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

function parsePositiveInt(value: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Expected a positive integer, received "${value}".`);
  }
  return parsed;
}

function parseNumber(value: string): number {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Expected a number, received "${value}".`);
  }
  return parsed;
}

function parseEmbeddingProvider(value: string): EmbeddingProvider {
  const provider = value.toLowerCase();
  if (provider === "local" || provider === "openai") {
    return provider;
  }
  throw new Error(`Unsupported embedding provider "${value}". Use local or openai.`);
}
