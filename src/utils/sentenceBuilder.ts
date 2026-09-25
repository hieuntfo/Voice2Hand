import { RecognitionResult } from '../types';
import { CONFIDENCE_THRESHOLD } from './knnClassifier';

export const STABILITY_FRAMES_REQUIRED = 8; // Stable for 8 frames
export const COOLDOWN_MS = 1000; // 1 second cooldown after commit
export const REST_FRAMES_REQUIRED = 5; // 5 frames of no hand detected counts as a rest

export class SentenceAggregator {
  private consecutiveLabel: string = '';
  private consecutiveCount: number = 0;
  private consecutiveConfidences: number[] = [];

  private lastCommittedLabel: string = '';
  private lastCommitTimestamp: number = 0;
  private noHandFrameCount: number = 0;
  private hasRestedSinceLastCommit: boolean = true;

  /**
   * Processes a new frame recognition result.
   * Returns newly committed token if conditions are met, otherwise null.
   */
  public processFrame(result: RecognitionResult): {
    committedToken: string | null;
    progress: number; // 0 to 1 progress towards confirming current gesture
    isCoolingDown: boolean;
    remainingCooldownMs: number;
  } {
    const now = Date.now();
    const timeSinceLastCommit = now - this.lastCommitTimestamp;
    const isCoolingDown = timeSinceLastCommit < COOLDOWN_MS;
    const remainingCooldownMs = isCoolingDown ? COOLDOWN_MS - timeSinceLastCommit : 0;

    // Track rest between hand gestures (no hands detected)
    if (result.handCount === 0) {
      this.noHandFrameCount++;
      if (this.noHandFrameCount >= REST_FRAMES_REQUIRED) {
        this.hasRestedSinceLastCommit = true;
      }
      this.consecutiveLabel = '';
      this.consecutiveCount = 0;
      this.consecutiveConfidences = [];
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown,
        remainingCooldownMs,
      };
    }

    this.noHandFrameCount = 0;

    // If cooldown is still active, don't accumulate confirmation frames yet
    if (isCoolingDown) {
      this.consecutiveLabel = '';
      this.consecutiveCount = 0;
      this.consecutiveConfidences = [];
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown: true,
        remainingCooldownMs,
      };
    }

    // Must be a confident recognized label
    if (!result.isConfident || !result.label) {
      this.consecutiveLabel = '';
      this.consecutiveCount = 0;
      this.consecutiveConfidences = [];
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown: false,
        remainingCooldownMs: 0,
      };
    }

    // Condition (c) check:
    // Label must be different from last committed label OR hand has rested between commits
    const canCommitThisLabel =
      result.label !== this.lastCommittedLabel || this.hasRestedSinceLastCommit;

    if (!canCommitThisLabel) {
      // User is holding same sign without resting hand
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown: false,
        remainingCooldownMs: 0,
      };
    }

    // Check consecutive frames with the same label
    if (result.label === this.consecutiveLabel) {
      this.consecutiveCount++;
      this.consecutiveConfidences.push(result.confidence);
    } else {
      this.consecutiveLabel = result.label;
      this.consecutiveCount = 1;
      this.consecutiveConfidences = [result.confidence];
    }

    const progress = Math.min(1, this.consecutiveCount / STABILITY_FRAMES_REQUIRED);

    // Condition (a): stable for at least N consecutive frames
    if (this.consecutiveCount >= STABILITY_FRAMES_REQUIRED) {
      // Condition (b): average confidence in that window >= threshold
      const avgConfidence =
        this.consecutiveConfidences.reduce((a, b) => a + b, 0) /
        this.consecutiveConfidences.length;

      if (avgConfidence >= CONFIDENCE_THRESHOLD) {
        const tokenToCommit = this.consecutiveLabel;
        this.lastCommittedLabel = tokenToCommit;
        this.lastCommitTimestamp = now;
        this.hasRestedSinceLastCommit = false;

        // Reset buffer
        this.consecutiveLabel = '';
        this.consecutiveCount = 0;
        this.consecutiveConfidences = [];

        return {
          committedToken: tokenToCommit,
          progress: 1,
          isCoolingDown: true,
          remainingCooldownMs: COOLDOWN_MS,
        };
      }
    }

    return {
      committedToken: null,
      progress,
      isCoolingDown: false,
      remainingCooldownMs: 0,
    };
  }

  public reset(): void {
    this.consecutiveLabel = '';
    this.consecutiveCount = 0;
    this.consecutiveConfidences = [];
    this.lastCommittedLabel = '';
    this.lastCommitTimestamp = 0;
    this.noHandFrameCount = 0;
    this.hasRestedSinceLastCommit = true;
  }
}
