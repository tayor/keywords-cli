import type { ClusterAssignment, ClusterResult, PhraseRecord } from "./types.js";
import { extractKeyphrases } from "./extraction.js";
import { cosineSimilarity, makeTermVector } from "./text.js";

export interface ClusterOptions {
  minSimilarity?: number;
  minClusterSize?: number;
}

export function clusterRecords(records: PhraseRecord[], options: ClusterOptions = {}): ClusterResult {
  const minSimilarity = options.minSimilarity ?? 0.32;
  const minClusterSize = Math.max(1, options.minClusterSize ?? 2);
  const phrases = records.map((record) => record.phrase);
  const vectors = phrases.map(makeTermVector);
  const graph = new Map<number, Set<number>>();

  for (let index = 0; index < phrases.length; index += 1) {
    graph.set(index, new Set());
  }

  for (let left = 0; left < phrases.length; left += 1) {
    for (let right = left + 1; right < phrases.length; right += 1) {
      const similarity = cosineSimilarity(vectors[left] ?? new Map(), vectors[right] ?? new Map());
      if (similarity >= minSimilarity) {
        graph.get(left)?.add(right);
        graph.get(right)?.add(left);
      }
    }
  }

  const visited = new Set<number>();
  const assignments: ClusterAssignment[] = [];
  const summary: ClusterResult["summary"] = [];
  let nextClusterId = 0;

  for (let index = 0; index < phrases.length; index += 1) {
    if (visited.has(index)) {
      continue;
    }
    const component = collectComponent(index, graph, visited);
    const clusterId = component.length >= minClusterSize ? nextClusterId++ : -1;
    const componentPhrases = component.map((componentIndex) => phrases[componentIndex] ?? "");
    const topic = clusterId === -1 ? "Unclustered" : labelCluster(componentPhrases, clusterId);

    for (const componentIndex of component) {
      assignments.push({
        phrase: phrases[componentIndex] ?? "",
        cluster: clusterId,
        topic,
        size: component.length
      });
    }

    if (clusterId >= 0) {
      summary.push({
        cluster: clusterId,
        topic,
        size: component.length,
        examples: componentPhrases.slice(0, 5)
      });
    }
  }

  assignments.sort((left, right) => left.cluster - right.cluster || left.phrase.localeCompare(right.phrase));
  summary.sort((left, right) => right.size - left.size || left.cluster - right.cluster);

  return { assignments, summary };
}

function collectComponent(start: number, graph: Map<number, Set<number>>, visited: Set<number>): number[] {
  const stack = [start];
  const component: number[] = [];
  visited.add(start);

  while (stack.length > 0) {
    const current = stack.pop();
    if (current === undefined) {
      continue;
    }
    component.push(current);
    for (const neighbor of graph.get(current) ?? []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        stack.push(neighbor);
      }
    }
  }

  return component;
}

function labelCluster(phrases: string[], clusterId: number): string {
  const [best] = extractKeyphrases(phrases, { topN: 1, minNgram: 1, maxNgram: 3, diversity: 0.4 });
  return best?.phrase ?? `cluster ${clusterId}`;
}
