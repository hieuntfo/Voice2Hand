import { SignSample, KNNMatch, RecognitionResult } from '../types';
import {
  computeHandVectorDistance,
  computeCosineSimilarity,
  distanceToSimilarity,
  FingerStates,
} from './handNormalization';

export const CONFIDENCE_THRESHOLD = 0.52; // 52% default for fast & accessible recognition
export const DEFAULT_K = 3;

export interface FrameInputData {
  handCount: number; // 0, 1, or 2
  hand1Points?: number[]; // 63 values
  hand2Points?: number[]; // 63 values
  fingerStates1?: FingerStates;
  fingerStates2?: FingerStates;
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
 * Computes single hand similarity evaluating both direct and mirrored orientation.
 */
function matchSingleHand(observed: number[], template: number[]): number {
  const distDirect = computeHandVectorDistance(observed, template);
  const cosDirect = computeCosineSimilarity(observed, template);
  const simDirect = 0.55 * distanceToSimilarity(distDirect) + 0.45 * Math.max(0, cosDirect);

  const flipped = flipHandX(observed);
  const distFlipped = computeHandVectorDistance(flipped, template);
  const cosFlipped = computeCosineSimilarity(flipped, template);
  const simFlipped = 0.55 * distanceToSimilarity(distFlipped) + 0.45 * Math.max(0, cosFlipped);

  return Math.max(simDirect, simFlipped);
}

/**
 * Calculates similarity between observed frame and a stored sign sample.
 * Combines 63D landmark distance with anatomical finger state constraints.
 */
export function computeSampleSimilarity(frame: FrameInputData, sample: SignSample): number {
  if (frame.handCount === 0 || !frame.hand1Points || !sample.hand1) {
    return 0;
  }

  let baseSimilarity = 0;

  if (frame.handCount === 1 && sample.handCount === 1) {
    baseSimilarity = matchSingleHand(frame.hand1Points, sample.hand1);
  } else if (frame.handCount === 2 && sample.handCount === 2) {
    if (!frame.hand2Points || !sample.hand2) return 0;
    // Test both direct and swapped
    const simDirect1 = matchSingleHand(frame.hand1Points, sample.hand1);
    const simDirect2 = matchSingleHand(frame.hand2Points, sample.hand2);
    const scoreDirect = (simDirect1 + simDirect2) / 2;

    const simSwapped1 = matchSingleHand(frame.hand1Points, sample.hand2);
    const simSwapped2 = matchSingleHand(frame.hand2Points, sample.hand1);
    const scoreSwapped = (simSwapped1 + simSwapped2) / 2;

    baseSimilarity = Math.max(scoreDirect, scoreSwapped);
  } else if (frame.handCount === 1 && sample.handCount === 2) {
    // Graceful single-hand signing for 2-handed gestures (like BÂY_GIỜ, VUI_VẺ)
    const match1 = matchSingleHand(frame.hand1Points, sample.hand1);
    const match2 = sample.hand2 ? matchSingleHand(frame.hand1Points, sample.hand2) : 0;
    baseSimilarity = Math.max(match1, match2) * 0.90;
  } else if (frame.handCount === 2 && sample.handCount === 1) {
    // User raised 2 hands for a 1-hand sign; match dominant hand
    const match1 = matchSingleHand(frame.hand1Points, sample.hand1);
    const match2 = frame.hand2Points ? matchSingleHand(frame.hand2Points, sample.hand1) : 0;
    baseSimilarity = Math.max(match1, match2) * 0.92;
  }

  // Apply anatomical finger state heuristics if available
  const f1 = frame.fingerStates1;
  if (f1) {
    const isThumbsUp = f1.thumbUp || (f1.thumb > 0.7 && f1.index < 0.35 && f1.middle < 0.35 && f1.ring < 0.35 && f1.pinky < 0.35);
    const isSingleFingerPoint = f1.index > 0.65 && f1.middle < 0.35 && f1.ring < 0.35 && f1.pinky < 0.35;
    const isAllFingersOpen = f1.index > 0.65 && f1.middle > 0.65 && f1.ring > 0.65 && f1.pinky > 0.65;
    const isFlatPalmTight = isAllFingersOpen && f1.spread < 0.28;
    const isSpreadWide = isAllFingersOpen && f1.spread >= 0.28;

    // 1. KHỎE: Only thumb extended, all other fingers closed into a fist
    if (isThumbsUp) {
      if (sample.label === 'KHỎE') {
        baseSimilarity = Math.max(baseSimilarity, 0.88) * 1.15;
      } else {
        baseSimilarity *= 0.15; // Suppress open-hand false positives like TẠM_BIỆT
      }
    }

    // 2. BẠN / TÔI: Single index finger pointing
    if (isSingleFingerPoint) {
      if (sample.label === 'BẠN' || sample.label === 'TÔI') {
        baseSimilarity = Math.max(baseSimilarity, 0.86) * 1.12;
      } else {
        baseSimilarity *= 0.15; // Suppress other gestures
      }
    }

    // 3. CẢM_ƠN: Fingers flat together
    if (isFlatPalmTight) {
      if (sample.label === 'CẢM_ƠN') {
        baseSimilarity = Math.max(baseSimilarity, 0.86) * 1.10;
      }
    }

    // 4. TẠM_BIỆT vs XIN_CHÀO: Open fingers spread
    if (isSpreadWide) {
      if (sample.label === 'TẠM_BIỆT' || sample.label === 'XIN_CHÀO') {
        baseSimilarity = Math.max(baseSimilarity, 0.85);
      } else if (sample.label === 'KHỎE' || sample.label === 'BẠN' || sample.label === 'TÔI') {
        baseSimilarity *= 0.20; // An open hand cannot be a thumbs-up or point
      }
    }
  }

  return Math.max(0, Math.min(1, baseSimilarity));
}

/**
 * Classifies an incoming frame of hand landmarks against the active dataset using k-NN.
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
    const blended = avgSim * 0.65 + stat.maxSim * 0.35;
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
