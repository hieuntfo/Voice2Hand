import { RecognitionResult } from '../types';

export interface AggregatorConfig {
  stabilityFrames: number;      // e.g. 4 frames (default, was 8)
  cooldownMs: number;           // e.g. 400ms (default, was 1000ms)
  restFramesRequired: number;   // e.g. 2 frames (default, was 5)
  confidenceThreshold: number;  // e.g. 0.52 (default, was 0.60)
}

export const DEFAULT_AGGREGATOR_CONFIG: AggregatorConfig = {
  stabilityFrames: 4,          // Super responsive (approx 150ms at 25-30fps)
  cooldownMs: 420,             // Fast recovery before next sign
  restFramesRequired: 2,       // Fast hand-rest detection
  confidenceThreshold: 0.52,   // Accessible threshold
};

export class SentenceAggregator {
  private config: AggregatorConfig;

  // Sliding history window for robust detection
  private recentHistory: Array<{ label: string; confidence: number; timestamp: number }> = [];
  private maxHistorySize: number = 6;

  private lastCommittedLabel: string = '';
  private lastCommitTimestamp: number = 0;
  private noHandFrameCount: number = 0;
  private hasRestedSinceLastCommit: boolean = true;

  constructor(customConfig?: Partial<AggregatorConfig>) {
    this.config = { ...DEFAULT_AGGREGATOR_CONFIG, ...customConfig };
  }

  public updateConfig(newConfig: Partial<AggregatorConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  public getConfig(): AggregatorConfig {
    return { ...this.config };
  }

  /**
   * Processes a new frame recognition result with sliding window aggregation.
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
    const isCoolingDown = timeSinceLastCommit < this.config.cooldownMs;
    const remainingCooldownMs = isCoolingDown ? this.config.cooldownMs - timeSinceLastCommit : 0;

    // 1. Hand presence tracking & rest detection
    if (result.handCount === 0) {
      this.noHandFrameCount++;
      if (this.noHandFrameCount >= this.config.restFramesRequired) {
        this.hasRestedSinceLastCommit = true;
      }
      this.recentHistory = [];
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown,
        remainingCooldownMs,
      };
    }

    this.noHandFrameCount = 0;

    // 2. Cooldown active: user just committed a sign, waiting for transition
    if (isCoolingDown) {
      this.recentHistory = [];
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown: true,
        remainingCooldownMs,
      };
    }

    // 3. Must have a valid recognized label with confidence >= threshold
    if (!result.label || result.confidence < this.config.confidenceThreshold) {
      // Degrade history gracefully instead of instantly wiping out
      if (this.recentHistory.length > 0) {
        this.recentHistory.shift();
      }
      const currentCount = this.recentHistory.length;
      return {
        committedToken: null,
        progress: Math.min(0.9, currentCount / this.config.stabilityFrames),
        isCoolingDown: false,
        remainingCooldownMs: 0,
      };
    }

    // 4. Same-label repetition protection:
    // Label can be committed if:
    // (a) It differs from the last committed label, OR
    // (b) Hands have rested (dropped/left frame) since the last commit
    const canCommitThisLabel =
      result.label !== this.lastCommittedLabel || this.hasRestedSinceLastCommit;

    if (!canCommitThisLabel) {
      return {
        committedToken: null,
        progress: 0,
        isCoolingDown: false,
        remainingCooldownMs: 0,
      };
    }

    // 5. Append to sliding history
    this.recentHistory.push({
      label: result.label,
      confidence: result.confidence,
      timestamp: now,
    });

    if (this.recentHistory.length > this.maxHistorySize) {
      this.recentHistory.shift();
    }

    // Count occurrences of candidate labels in recent history
    const counts: Record<string, { count: number; totalConf: number }> = {};
    for (const item of this.recentHistory) {
      if (!counts[item.label]) {
        counts[item.label] = { count: 0, totalConf: 0 };
      }
      counts[item.label].count++;
      counts[item.label].totalConf += item.confidence;
    }

    let dominantLabel = '';
    let maxCount = 0;
    let avgConf = 0;

    for (const [lbl, data] of Object.entries(counts)) {
      if (data.count > maxCount) {
        maxCount = data.count;
        dominantLabel = lbl;
        avgConf = data.totalConf / data.count;
      }
    }

    const progress = Math.min(1, maxCount / this.config.stabilityFrames);

    // 6. Verification: has dominant label reached stabilityFrames requirement?
    if (
      maxCount >= this.config.stabilityFrames &&
      avgConf >= this.config.confidenceThreshold &&
      dominantLabel
    ) {
      const tokenToCommit = dominantLabel;
      this.lastCommittedLabel = tokenToCommit;
      this.lastCommitTimestamp = now;
      this.hasRestedSinceLastCommit = false;
      this.recentHistory = []; // Reset history after commit

      return {
        committedToken: tokenToCommit,
        progress: 1,
        isCoolingDown: true,
        remainingCooldownMs: this.config.cooldownMs,
      };
    }

    return {
      committedToken: null,
      progress,
      isCoolingDown: false,
      remainingCooldownMs: 0,
    };
  }

  public reset(): void {
    this.recentHistory = [];
    this.lastCommittedLabel = '';
    this.lastCommitTimestamp = 0;
    this.noHandFrameCount = 0;
    this.hasRestedSinceLastCommit = true;
  }
}
