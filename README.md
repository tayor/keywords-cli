# keywords-cli [WIP]

`keywords-cli` is a TypeScript CLI for phrase research. You can use it for keywords, niche ideas, product ideas, customer-support topics, app-store terms, feature requests, research themes, or any phrase list with a positive metric and a friction metric.

## Install and Run

From this folder during development:

```bash
npm install
npm run build
npm link
keywords-cli --help
```

After publishing to npm:

```bash
npx keywords-cli analyze keywords.csv --value-column "Search Volume" --friction-column "Ranking Difficulty"
npm i -g keywords-cli
keywords-cli analyze ideas.csv --text-column idea --value-column impact --friction-column effort
```

The package exposes the `keywords-cli` binary.

## Commands

```bash
keywords-cli analyze <input>
keywords-cli extract <input>
keywords-cli cluster <input>
keywords-cli questions <input>
keywords-cli gap --mine mine.txt --competitor competitor.txt
keywords-cli train <input> --model model.json
keywords-cli predict <input> --model model.json
keywords-cli brief "topic" --related "phrase one,phrase two"
keywords-cli discover "broad domain" --ai
```

Inputs can be CSV, TSV, JSON, TXT, stdin (`-`), or raw newline-separated text. For structured files, the CLI auto-detects common phrase columns such as `Keyword`, `query`, `term`, `idea`, `niche`, `title`, and `text`. You can override this with `--text-column`.

## Opportunity Scoring

The base opportunity target follows the original value/friction shape: normalize the positive metric, divide by normalized friction with a small epsilon, then add light text signals for specificity and intent:

```bash
keywords-cli analyze keywords.csv \
  --value-column "Search Volume" \
  --friction-column "Ranking Difficulty"
```

For non-SEO workflows, use your own metric names:

```bash
keywords-cli analyze product-ideas.csv \
  --text-column idea \
  --value-column impact \
  --friction-column effort
```

The model treats the value column as higher-is-better and the friction column as lower-is-better. If one or both columns are missing, it falls back to text-only proxy signals such as specificity, intent, and broadness.

## Embeddings + LGBM Model

Train an embedding-backed LGBM-style boosted-tree model and use it against new phrase lists:

```bash
keywords-cli train historical-keywords.csv \
  --value-column "Search Volume" \
  --friction-column "Ranking Difficulty" \
  --embedding-provider openai \
  --model opportunity-model.json

keywords-cli predict new-keywords.txt --model opportunity-model.json
```

Use `--embedding-provider openai` with `OPENAI_API_KEY` for direct OpenAI embeddings. The offline default is `--embedding-provider local`, which creates deterministic local embeddings for tests and no-key workflows while keeping the saved model shape as embeddings plus LGBM-style trees.

## AI-Assisted Commands

`brief` and `discover` are deterministic by default. Add `--ai` and set `OPENAI_API_KEY` to call OpenAI Chat Completions directly:

```bash
export OPENAI_API_KEY="..."
keywords-cli brief "ai coloring book prompts" --ai
keywords-cli discover "sustainable living products" --ai --count 20
```

No API key is saved by the CLI.

## Output Formats

Most commands support:

```bash
--format table
--format json
--format csv
--format markdown
--out results.json
```
