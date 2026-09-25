export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export type HandLandmarks = Landmark[];

export interface NormalizedHandData {
  points: number[]; // 63 values: 21 points * (x, y, z)
  scale: number;
  wrist: Landmark;
}

export interface SignSample {
  id: string;
  label: string; // Convention: UPPERCASE_WITH_UNDERSCORE, e.g. "TÔI", "BÂY_GIỜ", "CẢM_THẤY", "VUI_VẺ"
  profile: string; // e.g. "khang", "lan", "minh"
  handCount: 1 | 2;
  hand1: number[]; // 63 values
  hand2?: number[]; // 63 values if handCount === 2
  interWristOffset?: {
    dx: number;
    dy: number;
    dz: number;
  };
  createdAt: number;
  notes?: string;
}

export interface KNNMatch {
  sample: SignSample;
  similarity: number; // 0 to 1
}

export interface RecognitionResult {
  label: string;
  confidence: number; // 0 to 1
  handCount: number;
  isConfident: boolean; // confidence >= CONFIDENCE_THRESHOLD
  statusText: string;
  topMatches: Array<{ label: string; score: number }>;
}

export interface SentenceCommit {
  id: string;
  token: string;
  timestamp: number;
  confidence: number;
}
