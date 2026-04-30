import { writeFile } from "node:fs/promises";
import type { OutputFormat } from "./types.js";

export async function emitOutput(data: unknown, format: OutputFormat, outPath?: string): Promise<string> {
  const rendered = renderOutput(data, format);
  if (outPath) {
    await writeFile(outPath, rendered, "utf8");
  }
  return rendered;
}

export function renderOutput(data: unknown, format: OutputFormat): string {
  if (format === "json") {
    return `${JSON.stringify(data, null, 2)}\n`;
  }

  const rows = Array.isArray(data) ? data : objectToRows(data);
  if (format === "csv") {
    return rowsToCsv(rows);
  }
  if (format === "markdown") {
    return rowsToMarkdown(rows);
  }
  return rowsToTable(rows);
}

export function rowsToCsv(rows: Array<Record<string, unknown>>): string {
  if (rows.length === 0) {
    return "";
  }
  const headers = collectHeaders(rows);
  const lines = [headers.map(escapeCsv).join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => escapeCsv(formatCell(row[header]))).join(","));
  }
  return `${lines.join("\n")}\n`;
}

export function rowsToMarkdown(rows: Array<Record<string, unknown>>): string {
  if (rows.length === 0) {
    return "No rows.\n";
  }
  const headers = collectHeaders(rows);
  const headerLine = `| ${headers.join(" | ")} |`;
  const separator = `| ${headers.map(() => "---").join(" | ")} |`;
  const body = rows.map((row) => `| ${headers.map((header) => escapeMarkdown(formatCell(row[header]))).join(" | ")} |`);
  return `${[headerLine, separator, ...body].join("\n")}\n`;
}

export function rowsToTable(rows: Array<Record<string, unknown>>): string {
  if (rows.length === 0) {
    return "No rows.\n";
  }
  const headers = collectHeaders(rows);
  const stringRows = rows.map((row) => headers.map((header) => truncate(formatCell(row[header]), 80)));
  const widths = headers.map((header, index) => {
    const maxCell = Math.max(...stringRows.map((row) => row[index]?.length ?? 0));
    return Math.min(80, Math.max(header.length, maxCell));
  });

  const header = headers.map((name, index) => name.padEnd(widths[index] ?? name.length)).join("  ");
  const rule = widths.map((width) => "-".repeat(width)).join("  ");
  const body = stringRows.map((row) => row.map((cell, index) => cell.padEnd(widths[index] ?? cell.length)).join("  "));
  return `${[header, rule, ...body].join("\n")}\n`;
}

export function normalizeFormat(value: string | undefined): OutputFormat {
  const format = (value ?? "table").toLowerCase();
  if (["table", "json", "csv", "markdown"].includes(format)) {
    return format as OutputFormat;
  }
  throw new Error(`Unsupported format "${value}". Use table, json, csv, or markdown.`);
}

function collectHeaders(rows: Array<Record<string, unknown>>): string[] {
  const headers: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!headers.includes(key)) {
        headers.push(key);
      }
    }
  }
  return headers;
}

function objectToRows(data: unknown): Array<Record<string, unknown>> {
  if (!data || typeof data !== "object") {
    return [{ value: data }];
  }
  const object = data as Record<string, unknown>;
  return Object.entries(object).map(([key, value]) => ({ key, value }));
}

function formatCell(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 100) / 100);
  }
  if (Array.isArray(value)) {
    return value.join("; ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function escapeCsv(value: unknown): string {
  const text = String(value ?? "");
  if (/[",\n\r]/u.test(text)) {
    return `"${text.replace(/"/gu, "\"\"")}"`;
  }
  return text;
}

function escapeMarkdown(value: string): string {
  return value.replace(/\|/gu, "\\|").replace(/\n/gu, "<br>");
}

function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, Math.max(0, maxLength - 1))}...`;
}
