import { Landmark, NormalizedHandData } from '../types';

export interface FingerStates {
  thumb: number;   // 0 = curled, 1 = extended
  index: number;   // 0 = curled, 1 = extended
  middle: number;  // 0 = curled, 1 = extended
  ring: number;    // 0 = curled, 1 = extended
  pinky: number;   // 0 = curled, 1 = extended
  thumbUp: boolean; // is thumb pointing upwards?
  spread: number;  // average distance between adjacent fingertips
}

export interface EnhancedHandFeatures {
  points: number[];        // 63 normalized coordinates
  scale: number;
  wrist: Landmark;
  fingerStates: FingerStates;
}

/**
 * Normalizes 21 3D landmarks for a single hand and extracts key finger articulation features.
 */
export function normalizeHandLandmarks(landmarks: Landmark[]): EnhancedHandFeatures | null {
  if (!landmarks || landmarks.length !== 21) {
    return null;
  }

  const wrist = landmarks[0];
  const midMcp = landmarks[9];

  // Scale: Distance from wrist to middle finger MCP
  const dx9 = midMcp.x - wrist.x;
  const dy9 = midMcp.y - wrist.y;
  const dz9 = (midMcp.z ?? 0) - (wrist.z ?? 0);
  const scale = Math.sqrt(dx9 * dx9 + dy9 * dy9 + dz9 * dz9) || 1.0;

  // 63 normalized points relative to wrist and scaled
  const points: number[] = [];
  for (let i = 0; i < 21; i++) {
    const p = landmarks[i];
    const dx = (p.x - wrist.x) / scale;
    const dy = (p.y - wrist.y) / scale;
    const dz = ((p.z ?? 0) - (wrist.z ?? 0)) / scale;
    points.push(dx, dy, dz);
  }

  // Calculate finger extension states based on tip-to-wrist vs mcp-to-wrist ratios
  const dist = (idxA: number, idxB: number) => {
    const pA = landmarks[idxA];
    const pB = landmarks[idxB];
    const dx = pA.x - pB.x;
    const dy = pA.y - pB.y;
    const dz = (pA.z ?? 0) - (pB.z ?? 0);
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  // Finger tips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
  // Finger MCPs: 2, 5, 9, 13, 17
  // Finger PIPs: 3, 6, 10, 14, 18
  const checkExt = (tipIdx: number, pipIdx: number, mcpIdx: number) => {
    const dTipWrist = dist(tipIdx, 0);
    const dPipWrist = dist(pipIdx, 0);
    const dMcpWrist = dist(mcpIdx, 0);
    if (dTipWrist > dMcpWrist * 1.15 && dTipWrist > dPipWrist * 1.05) return 1.0;
    if (dTipWrist < dMcpWrist * 0.95) return 0.0;
    return Math.max(0, Math.min(1, (dTipWrist - dMcpWrist * 0.95) / (dMcpWrist * 0.2)));
  };

  // Thumb special check
  const dThumbTipWrist = dist(4, 0);
  const dThumbMcpWrist = dist(2, 0);
  const dThumbTipIndexMcp = dist(4, 5);
  const thumbExt = (dThumbTipWrist > dThumbMcpWrist * 1.1 && dThumbTipIndexMcp > scale * 0.45) ? 1.0 : 0.0;

  // Thumb pointing up check: tip is well above wrist and above thumb MCP
  const thumbUp = (landmarks[4].y < wrist.y - scale * 0.5) && (landmarks[4].y < landmarks[2].y);

  const indexExt = checkExt(8, 6, 5);
  const middleExt = checkExt(12, 10, 9);
  const ringExt = checkExt(16, 14, 13);
  const pinkyExt = checkExt(20, 18, 17);

  // Spread between adjacent fingertips
  const s1 = dist(8, 12) / scale;
  const s2 = dist(12, 16) / scale;
  const s3 = dist(16, 20) / scale;
  const avgSpread = (s1 + s2 + s3) / 3;

  const fingerStates: FingerStates = {
    thumb: thumbExt,
    index: indexExt,
    middle: middleExt,
    ring: ringExt,
    pinky: pinkyExt,
    thumbUp,
    spread: avgSpread,
  };

  return {
    points,
    scale,
    wrist,
    fingerStates,
  };
}

/**
 * Computes Euclidean distance between two normalized 63-dimensional hand vectors.
 */
export function computeHandVectorDistance(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 999;
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
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
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
 */
export function distanceToSimilarity(dist: number): number {
  const maxAcceptableDist = 0.62;
  if (dist >= maxAcceptableDist) return 0;
  const rawScore = 1 - dist / maxAcceptableDist;
  return Math.max(0, Math.min(1, Math.pow(rawScore, 0.70)));
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
