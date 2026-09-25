import { SignSample, KNNMatch, RecognitionResult } from '../types';
import {
  computeHandVectorDistance,
  computeCosineSimilarity,
  distanceToSimilarity,
} from './handNormalization';

export const CONFIDENCE_THRESHOLD = 0.52; // 52% default for fast & accessible recognition
export const DEFAULT_K = 3;

export interface FrameInputData {
  handCount: number; // 0, 1, or 2
  hand1Points?: number[]; // 63 values
  hand2Points?: number[]; // 63 values
  interWristOffset?: { dx: number; dy: number; dz: number };
}

/**
 * Creates horizontally flipped vector (negates relative x coordinate)
 * to support both left and right hand signing seamlessly.
 */
function flipHandX(points: number[]): number[] {
  const flipped = [...points];
  for (let i = 0; i < 21; i++) {
    flipped[i * 3] = -flipped[i * 3]; // Invert dx
  }
  return flipped;
}

/**
 * Calculates similarity between a current observed frame and a stored sign sample.
 * Evaluates both direct and horizontally mirrored hand for 1-hand gestures
 * so both left-handed and right-handed signers get instant high confidence.
 */
export function computeSampleSimilarity(frame: FrameInputData, sample: SignSample): number {
  if (frame.handCount !== sample.handCount) {
    return 0;
  }

  if (frame.handCount === 1) {
    if (!frame.hand1Points || !sample.hand1) return 0;

    // Test direct orientation
    const distDirect = computeHandVectorDistance(frame.hand1Points, sample.hand1);
    const cosDirect = computeCosineSimilarity(frame.hand1Points, sample.hand1);
    const simDirect = 0.60 * distanceToSimilarity(distDirect) + 0.40 * Math.max(0, cosDirect);

    // Test mirrored (left vs right hand) orientation
    const flippedPoints = flipHandX(frame.hand1Points);
    const distFlipped = computeHandVectorDistance(flippedPoints, sample.hand1);
    const cosFlipped = computeCosineSimilarity(flippedPoints, sample.hand1);
    const simFlipped = 0.60 * distanceToSimilarity(distFlipped) + 0.40 * Math.max(0, cosFlipped);

    return Math.max(0, Math.min(1, Math.max(simDirect, simFlipped)));
  }

  if (frame.handCount === 2) {
    if (!frame.hand1Points || !frame.hand2Points || !sample.hand1 || !sample.hand2) return 0;

    // Direct orientation (h1->s1, h2->s2)
    const dist1 = computeHandVectorDistance(frame.hand1Points, sample.hand1);
    const dist2 = computeHandVectorDistance(frame.hand2Points, sample.hand2);
    const scoreDirect = (distanceToSimilarity(dist1) + distanceToSimilarity(dist2)) / 2;

    // Swapped orientation (h1->s2, h2->s1)
    const distSwapped1 = computeHandVectorDistance(frame.hand1Points, sample.hand2);
    const distSwapped2 = computeHandVectorDistance(frame.hand2Points, sample.hand1);
    const scoreSwapped = (distanceToSimilarity(distSwapped1) + distanceToSimilarity(distSwapped2)) / 2;

    return Math.max(scoreDirect, scoreSwapped);
  }

  return 0;
}

/**
 * Classifies an incoming frame of hand landmarks against the active dataset using k-NN.
 *
 * @param frame Observed hand landmarks in current video frame
 * @param dataset The full or profile-filtered list of SignSamples
 * @param activeProfile 'global' for Global Model, or profile name (e.g. 'khang')
 * @param k Number of nearest neighbors to aggregate (default 3)
 * @param threshold Confidence threshold to consider confident
 */
export function classifyHandGesture(
  frame: FrameInputData,
  dataset: SignSample[],
  activeProfile: string = 'global',
  k: number = DEFAULT_K,
  threshold: number = CONFIDENCE_THRESHOLD
): RecognitionResult {
  // If no hands detected
  if (frame.handCount === 0 || !frame.hand1Points) {
    return {
      label: '',
      confidence: 0,
      handCount: 0,
      isConfident: false,
      statusText: 'Đang chờ tay...',
      topMatches: [],
    };
  }

  // Filter dataset by profile if not 'global'
  const pool = activeProfile === 'global'
    ? dataset
    : dataset.filter((s) => s.profile.toLowerCase() === activeProfile.toLowerCase());

  if (pool.length === 0) {
    return {
      label: '',
      confidence: 0,
      handCount: frame.handCount,
      isConfident: false,
      statusText: activeProfile === 'global'
        ? 'Chưa có mẫu ký hiệu nào trong hệ thống'
        : `Chưa có mẫu nào cho Profile: ${activeProfile}`,
      topMatches: [],
    };
  }

  // Calculate similarity to all samples in the pool
  const matches: KNNMatch[] = pool.map((sample) => ({
    sample,
    similarity: computeSampleSimilarity(frame, sample),
  }));

  // Sort descending by similarity
  matches.sort((a, b) => b.similarity - a.similarity);

  // Group by label to find best candidate class
  const classScores: Record<string, { totalSim: number; count: number; maxSim: number }> = {};
  const effectiveK = Math.min(k, matches.length);
  const topKNeighbors = matches.slice(0, effectiveK);

  for (const m of topKNeighbors) {
    const lbl = m.sample.label;
    if (!classScores[lbl]) {
      classScores[lbl] = { totalSim: 0, count: 0, maxSim: 0 };
    }
    classScores[lbl].totalSim += m.similarity;
    classScores[lbl].count += 1;
    if (m.similarity > classScores[lbl].maxSim) {
      classScores[lbl].maxSim = m.similarity;
    }
  }

  // Find class with highest average similarity in top K
  let bestLabel = '';
  let bestConfidence = 0;

  for (const [lbl, stat] of Object.entries(classScores)) {
    const avgSim = stat.totalSim / stat.count;
    // Blend average similarity with top-1 similarity for stability
    const blended = avgSim * 0.7 + stat.maxSim * 0.3;
    if (blended > bestConfidence) {
      bestConfidence = blended;
      bestLabel = lbl;
    }
  }

  const isConfident = bestConfidence >= threshold && Boolean(bestLabel);
  const percent = Math.round(bestConfidence * 100);

  let statusText = '';
  if (!isConfident) {
    statusText = 'Không chắc chắn';
  } else {
    statusText = `✓ ${bestLabel} · confidence ${percent}%`;
  }

  // Generate top matches list for inspection / debugging
  const topSummary = Object.entries(classScores)
    .map(([lbl, stat]) => ({
      label: lbl,
      score: Math.round((stat.totalSim / stat.count) * 100) / 100,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  return {
    label: bestLabel,
    confidence: bestConfidence,
    handCount: frame.handCount,
    isConfident,
    statusText,
    topMatches: topSummary,
  };
}
