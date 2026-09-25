import { Landmark, NormalizedHandData } from '../types';

/**
 * Normalizes 21 3D landmarks for a single hand:
 * 1. Translates wrist (index 0) to (0, 0, 0).
 * 2. Scales coordinates by the distance between wrist (0) and middle finger MCP (9)
 *    so hand size & distance from camera are invariant.
 * Returns an array of 63 floats (21 points * 3 coordinates).
 */
export function normalizeHandLandmarks(landmarks: Landmark[]): NormalizedHandData | null {
  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  const wrist = landmarks[0];
  const midMcp = landmarks[9];

  // Vector from wrist to middle finger MCP
  const dx9 = midMcp.x - wrist.x;
  const dy9 = midMcp.y - wrist.y;
  const dz9 = (midMcp.z ?? 0) - (wrist.z ?? 0);
  const scale = Math.sqrt(dx9 * dx9 + dy9 * dy9 + dz9 * dz9) || 1.0;

  const points: number[] = [];
  for (let i = 0; i < 21; i++) {
    const p = landmarks[i];
    const dx = (p.x - wrist.x) / scale;
    const dy = (p.y - wrist.y) / scale;
    const dz = ((p.z ?? 0) - (wrist.z ?? 0)) / scale;
    points.push(dx, dy, dz);
  }

  return {
    points,
    scale,
    wrist,
  };
}

/**
 * Computes Euclidean distance between two normalized 63-dimensional hand vectors.
 */
export function computeHandVectorDistance(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 999;
  let sumSq = 0;
  for (let i = 0; i < vecA.length; i++) {
    const diff = vecA[i] - vecB[i];
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq / vecA.length);
}

/**
 * Cosine similarity between two vectors.
 */
export function computeCosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA <= 0 || normB <= 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Converts a Euclidean distance on normalized landmarks to a similarity score [0, 1].
 * For normalized hands, a distance under 0.12 is very close (similarity ~0.85-0.98).
 * A distance of 0.4 or higher is very different (similarity -> 0).
 */
export function distanceToSimilarity(dist: number): number {
  // Linear decay with sharp dropoff for dissimilar poses
  const maxAcceptableDist = 0.45;
  if (dist >= maxAcceptableDist) return 0;
  const rawScore = 1 - dist / maxAcceptableDist;
  // Non-linear enhancement so close matches get realistic confidence in 80-98% range
  return Math.max(0, Math.min(1, Math.pow(rawScore, 0.85)));
}

/**
 * MediaPipe Hand connections pairs for drawing landmarks.
 */
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm base
  [5, 9],
  [9, 13],
  [13, 17],
];
