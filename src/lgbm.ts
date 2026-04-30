import type { LightGbmTree } from "./types.js";

export interface LightGbmTrainOptions {
  trees?: number;
  learningRate?: number;
  minGain?: number;
}

export interface LightGbmRegressor {
  objective: "regression";
  featureCount: number;
  initialScore: number;
  learningRate: number;
  trees: LightGbmTree[];
}

export function trainLightGbmRegressor(
  features: number[][],
  targets: number[],
  options: LightGbmTrainOptions = {}
): { model: LightGbmRegressor; loss: number } {
  if (features.length !== targets.length || features.length === 0) {
    throw new Error("Training requires matching feature and target arrays.");
  }

  const treeCount = Math.max(1, options.trees ?? 80);
  const learningRate = options.learningRate ?? 0.08;
  const minGain = options.minGain ?? 1e-9;
  const featureCount = features[0]?.length ?? 0;
  const initialScore = average(targets);
  const predictions = Array.from({ length: targets.length }, () => initialScore);
  const trees: LightGbmTree[] = [];

  for (let treeIndex = 0; treeIndex < treeCount; treeIndex += 1) {
    const residuals = targets.map((target, index) => target - (predictions[index] ?? 0));
    const stump = findBestStump(features, residuals, featureCount);
    if (!stump || stump.gain < minGain) {
      break;
    }
    trees.push(stump);
    for (let rowIndex = 0; rowIndex < features.length; rowIndex += 1) {
      const row = features[rowIndex] ?? [];
      const leafValue = (row[stump.feature] ?? 0) <= stump.threshold ? stump.leftValue : stump.rightValue;
      predictions[rowIndex] = (predictions[rowIndex] ?? 0) + learningRate * leafValue;
    }
  }

  return {
    model: {
      objective: "regression",
      featureCount,
      initialScore,
      learningRate,
      trees
    },
    loss: meanSquaredError(predictions, targets)
  };
}

export function predictLightGbmRegressor(features: number[][], model: LightGbmRegressor): number[] {
  return features.map((row) => {
    let prediction = model.initialScore;
    for (const tree of model.trees) {
      const leafValue = (row[tree.feature] ?? 0) <= tree.threshold ? tree.leftValue : tree.rightValue;
      prediction += model.learningRate * leafValue;
    }
    return prediction;
  });
}

function findBestStump(features: number[][], residuals: number[], featureCount: number): LightGbmTree | undefined {
  const baselineError = sumSquared(residuals);
  let bestTree: LightGbmTree | undefined;

  for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
    const thresholds = candidateThresholds(features.map((row) => row[featureIndex] ?? 0));
    for (const threshold of thresholds) {
      const leftResiduals: number[] = [];
      const rightResiduals: number[] = [];

      for (let rowIndex = 0; rowIndex < features.length; rowIndex += 1) {
        const row = features[rowIndex] ?? [];
        if ((row[featureIndex] ?? 0) <= threshold) {
          leftResiduals.push(residuals[rowIndex] ?? 0);
        } else {
          rightResiduals.push(residuals[rowIndex] ?? 0);
        }
      }

      if (leftResiduals.length === 0 || rightResiduals.length === 0) {
        continue;
      }

      const leftValue = average(leftResiduals);
      const rightValue = average(rightResiduals);
      const error = sumSquared(leftResiduals.map((value) => value - leftValue)) + sumSquared(rightResiduals.map((value) => value - rightValue));
      const gain = baselineError - error;
      if (!bestTree || gain > bestTree.gain) {
        bestTree = { feature: featureIndex, threshold, leftValue, rightValue, gain };
      }
    }
  }

  return bestTree;
}

function candidateThresholds(values: number[]): number[] {
  const uniqueValues = [...new Set(values)].sort((left, right) => left - right);
  if (uniqueValues.length < 2) {
    return [];
  }

  const thresholds: number[] = [];
  const candidateCount = Math.min(16, uniqueValues.length - 1);
  for (let candidateIndex = 1; candidateIndex <= candidateCount; candidateIndex += 1) {
    const valueIndex = Math.min(uniqueValues.length - 2, Math.floor((candidateIndex * (uniqueValues.length - 1)) / (candidateCount + 1)));
    const left = uniqueValues[valueIndex] ?? 0;
    const right = uniqueValues[valueIndex + 1] ?? left;
    thresholds.push((left + right) / 2);
  }
  return [...new Set(thresholds)];
}

function sumSquared(values: number[]): number {
  return values.reduce((sum, value) => sum + value * value, 0);
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function meanSquaredError(predictions: number[], targets: number[]): number {
  if (predictions.length === 0) {
    return 0;
  }
  const total = predictions.reduce((sum, prediction, index) => {
    const error = prediction - (targets[index] ?? 0);
    return sum + error * error;
  }, 0);
  return total / predictions.length;
}