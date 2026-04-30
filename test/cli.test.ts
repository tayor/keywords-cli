import { execFile } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const rootDir = fileURLToPath(new URL("..", import.meta.url));
const cliPath = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
const tsxPath = fileURLToPath(new URL("../node_modules/tsx/dist/cli.mjs", import.meta.url));

async function runCli(args: string[]): Promise<{ stdout: string; stderr: string }> {
  const { stdout, stderr } = await execFileAsync(process.execPath, [tsxPath, cliPath, ...args], {
    cwd: rootDir,
    encoding: "utf8",
    maxBuffer: 1024 * 1024,
    timeout: 10_000
  });
  return { stdout: String(stdout), stderr: String(stderr) };
}

describe("cli binary", () => {
  it("prints the package version", async () => {
    const { stdout, stderr } = await runCli(["--version"]);

    expect(stderr).toBe("");
    expect(stdout.trim()).toBe("0.1.0");
  });

  it("analyzes CSV input as JSON", async () => {
    const tempDir = await mkdtemp(join(tmpdir(), "keywords-cli-"));
    const inputPath = join(tempDir, "keywords.csv");
    await writeFile(inputPath, "Keyword,Search Volume,Ranking Difficulty\nlow competition dog treats,1200,20\nbroad dog,5000,80\n", "utf8");

    const { stdout, stderr } = await runCli([
      "analyze",
      inputPath,
      "--value-column",
      "Search Volume",
      "--friction-column",
      "Ranking Difficulty",
      "--format",
      "json"
    ]);

    const rows = JSON.parse(stdout) as Array<{ phrase: string; source: string }>;
    expect(stderr).toBe("");
    expect(rows).toHaveLength(2);
    expect(rows.map((row) => row.phrase)).toContain("low competition dog treats");
    expect(rows.every((row) => row.source === "columns")).toBe(true);
  });
});