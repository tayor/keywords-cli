import { describe, expect, it } from "vitest";
import { clusterRecords, compareGaps, findQuestionRecords, predictWithModel, recordsFromDelimited, scoreRecords, trainOpportunityModel } from "../src/index.js";

function record(phrase: string, value?: number, friction?: number) {
  return {
    phrase,
    sourceIndex: 0,
    raw: { phrase, value, friction },
    ...(value === undefined ? {} : { value }),
    ...(friction === undefined ? {} : { friction })
  };
}

describe("input parsing", () => {
  it("auto-detects keyword, value, and friction columns", () => {
    const csv = `Keyword,Search Volume,Ranking Difficulty\nlow competition dog treats,1200,20\nbroad dog,5000,80\n`;
    const records = recordsFromDelimited(csv);

    expect(records).toHaveLength(2);
    expect(records[0]?.phrase).toBe("low competition dog treats");
    expect(records[0]?.value).toBe(1200);
    expect(records[0]?.friction).toBe(20);
  });

  it("does not mistake headerless CSV rows for detected headers", () => {
    const csv = `keyword research,100\nai coloring prompts,200\n`;
    const records = recordsFromDelimited(csv);

    expect(records.map((item) => item.phrase)).toEqual(["keyword research", "ai coloring prompts"]);
  });
});

describe("opportunity scoring", () => {
  it("prioritizes high value and low friction phrases", () => {
    const scored = scoreRecords([
      record("niche ai coloring prompts", 1000, 15),
      record("generic coloring", 2000, 90),
      record("tiny demand phrase", 50, 5)
    ]);

    expect(scored[0]?.phrase).toBe("niche ai coloring prompts");
    expect(scored[0]?.metricsSource).toBe("columns");
  });

  it("keeps the ratio-style scoring target for training", () => {
    const scored = scoreRecords([
      record("low competition dog treats", 1200, 20),
      record("broad dog", 5000, 80)
    ]);

    expect(scored[0]?.phrase).toBe("broad dog");
  });
});

describe("questions", () => {
  it("detects question phrases", () => {
    const questions = findQuestionRecords([
      record("how to validate a niche"),
      record("best niche research tools"),
      record("is keyword clustering useful?")
    ]);

    expect(questions.map((item) => item.phrase)).toEqual(["how to validate a niche", "is keyword clustering useful?"]);
  });
});

describe("clustering", () => {
  it("groups related phrases by local similarity", () => {
    const result = clusterRecords([
      record("email marketing automation"),
      record("email marketing tools"),
      record("project management software"),
      record("project management templates")
    ], { minSimilarity: 0.25, minClusterSize: 2 });

    const clustered = result.assignments.filter((item) => item.cluster >= 0);
    expect(new Set(clustered.map((item) => item.cluster)).size).toBe(2);
    expect(result.summary).toHaveLength(2);
  });
});

describe("gap analysis", () => {
  it("returns competitor-only opportunities", () => {
    const gap = compareGaps([
      record("email marketing tools"),
      record("keyword clustering")
    ], [
      record("email marketing tools"),
      record("ai keyword research")
    ]);

    expect(gap.shared).toEqual(["email marketing tools"]);
    expect(gap.uniqueToCompetitor).toEqual(["ai keyword research"]);
  });
});

describe("embedding lgbm model", () => {
  it("trains and predicts scores", async () => {
    const training = [
      record("best ai coloring book prompts", 2000, 20),
      record("coloring book", 6000, 95),
      record("how to draw flowers", 900, 40),
      record("printable mandala templates", 1400, 25)
    ];
    const model = await trainOpportunityModel(training, { trees: 12, dimensions: 64 });
    const predictions = await predictWithModel([
      record("ai coloring page prompts"),
      record("generic book")
    ], model);

    expect(model.modelType).toBe("embedding-lightgbm");
    expect(model.training.samples).toBe(4);
    expect(model.lgbm.trees.length).toBeGreaterThan(0);
    expect(predictions).toHaveLength(2);
    expect(predictions[0]?.predictedScore).toBeGreaterThanOrEqual(0);
  });
});
