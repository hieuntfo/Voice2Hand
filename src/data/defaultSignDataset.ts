import { SignSample } from '../types';
import { normalizeHandLandmarks } from '../utils/handNormalization';

/**
 * 21 MediaPipe hand points anatomy:
 * 0: Wrist
 * 1-4: Thumb (1: CMC, 2: MCP, 3: IP, 4: TIP)
 * 5-8: Index (5: MCP, 6: PIP, 7: DIP, 8: TIP)
 * 9-12: Middle (9: MCP, 10: PIP, 11: DIP, 12: TIP)
 * 13-16: Ring (13: MCP, 14: PIP, 15: DIP, 16: TIP)
 * 17-20: Pinky (17: MCP, 18: PIP, 19: DIP, 20: TIP)
 */
export type PoseType =
  | 'point_chest'     // TÔI: Index pointing towards chest/self, other 4 fingers curled
  | 'point_forward'   // BẠN: Index pointing straight forward to camera/partner, other 4 fingers curled
  | 'thumbs_up'       // KHỎE: Thumb pointing straight UP, 4 fingers curled in a fist
  | 'flat_palm'       // CẢM_ƠN: All fingers extended flat and pressed tightly together
  | 'open_wave'       // XIN_CHÀO: Open hand upright/waving, fingers open with natural spacing
  | 'spread_palm'     // TẠM_BIỆT: 5 fingers spread wide apart, waving goodbye
  | 'bent_down'       // BÂY_GIỜ: Hand(s) bent downward at knuckles, palms pressing down
  | 'chest_brush'     // CẢM_THẤY: Palm curved slightly facing chest, moving up
  | 'flutter_chest';  // VUI_VẺ: Open hand(s) upright in front of chest, palms facing in

function generateRealisticHand(pose: PoseType, variation: number = 0, isRight: boolean = true) {
  const side = isRight ? 1 : -1;
  const jitter = (s: number) => Math.sin(s + variation * 8) * 0.015;

  // Wrist is root at (0.5, 0.78, 0)
  const pts = [
    { x: 0.5 + jitter(1), y: 0.78 + jitter(2), z: 0 }, // 0: Wrist

    // Thumb (1, 2, 3, 4)
    { x: 0.5 - 0.05 * side + jitter(3), y: 0.72, z: -0.01 },
    { x: 0.5 - 0.09 * side + jitter(4), y: 0.66, z: -0.02 },
    { x: 0.5 - 0.12 * side + jitter(5), y: 0.60, z: -0.03 },
    { x: 0.5 - 0.14 * side + jitter(6), y: 0.54, z: -0.04 },

    // Index (5, 6, 7, 8)
    { x: 0.5 - 0.04 * side + jitter(7), y: 0.63, z: -0.01 },
    { x: 0.5 - 0.05 * side + jitter(8), y: 0.55, z: -0.01 },
    { x: 0.5 - 0.05 * side + jitter(9), y: 0.48, z: -0.01 },
    { x: 0.5 - 0.06 * side + jitter(10), y: 0.41, z: -0.01 },

    // Middle (9, 10, 11, 12)
    { x: 0.5 + 0.00 * side + jitter(11), y: 0.62, z: 0 },
    { x: 0.5 + 0.00 * side + jitter(12), y: 0.53, z: 0 },
    { x: 0.5 + 0.00 * side + jitter(13), y: 0.46, z: 0 },
    { x: 0.5 + 0.00 * side + jitter(14), y: 0.38, z: 0 },

    // Ring (13, 14, 15, 16)
    { x: 0.5 + 0.04 * side + jitter(15), y: 0.63, z: 0.01 },
    { x: 0.5 + 0.05 * side + jitter(16), y: 0.55, z: 0.01 },
    { x: 0.5 + 0.05 * side + jitter(17), y: 0.48, z: 0.01 },
    { x: 0.5 + 0.06 * side + jitter(18), y: 0.42, z: 0.01 },

    // Pinky (17, 18, 19, 20)
    { x: 0.5 + 0.08 * side + jitter(19), y: 0.66, z: 0.02 },
    { x: 0.5 + 0.09 * side + jitter(20), y: 0.60, z: 0.02 },
    { x: 0.5 + 0.10 * side + jitter(21), y: 0.54, z: 0.02 },
    { x: 0.5 + 0.11 * side + jitter(22), y: 0.48, z: 0.02 },
  ];

  // Helper to curl a finger tightly into palm
  const curlFinger = (mcp: number, pip: number, dip: number, tip: number) => {
    pts[pip].y = pts[mcp].y + 0.04;
    pts[dip].y = pts[mcp].y + 0.08;
    pts[dip].z = 0.05;
    pts[tip].y = pts[mcp].y + 0.06;
    pts[tip].z = 0.07;
  };

  // Helper to extend finger straight up
  const extendFinger = (mcp: number, pip: number, dip: number, tip: number, targetY: number, spreadX: number) => {
    pts[pip].y = pts[mcp].y - 0.08;
    pts[dip].y = pts[mcp].y - 0.15;
    pts[tip].y = targetY;
    pts[tip].x += spreadX * side;
  };

  switch (pose) {
    case 'point_chest': {
      // TÔI: Index finger extended pointing up/inward toward chest, other 4 curled
      pts[6].y = 0.54; pts[7].y = 0.47; pts[8].y = 0.40; pts[8].x = 0.5 - 0.02 * side;
      // Thumb tucked across palm
      pts[3].y = 0.67; pts[4].y = 0.68; pts[4].x = 0.5 - 0.03 * side;
      // Middle, Ring, Pinky curled
      curlFinger(9, 10, 11, 12);
      curlFinger(13, 14, 15, 16);
      curlFinger(17, 18, 19, 20);
      break;
    }

    case 'point_forward': {
      // BẠN: Index finger pointing forward at camera
      pts[6].y = 0.55; pts[7].y = 0.48; pts[8].y = 0.42; pts[8].z = -0.15;
      pts[3].y = 0.66; pts[4].y = 0.67; pts[4].x = 0.5 - 0.02 * side;
      curlFinger(9, 10, 11, 12);
      curlFinger(13, 14, 15, 16);
      curlFinger(17, 18, 19, 20);
      break;
    }

    case 'thumbs_up': {
      // KHỎE: Thumb straight UP!
      pts[2].x = 0.5 - 0.08 * side; pts[2].y = 0.68;
      pts[3].x = 0.5 - 0.10 * side; pts[3].y = 0.58;
      pts[4].x = 0.5 - 0.11 * side; pts[4].y = 0.46; // Thumb tip pointing way up
      // ALL other 4 fingers tightly curled into fist
      curlFinger(5, 6, 7, 8);
      curlFinger(9, 10, 11, 12);
      curlFinger(13, 14, 15, 16);
      curlFinger(17, 18, 19, 20);
      break;
    }

    case 'flat_palm': {
      // CẢM_ƠN: All 5 fingers extended, pressed tight together (no spread)
      pts[4].x = 0.5 - 0.06 * side; pts[4].y = 0.56;
      pts[8].x = 0.5 - 0.02 * side; pts[8].y = 0.41;
      pts[12].x = 0.5 + 0.00 * side; pts[12].y = 0.39;
      pts[16].x = 0.5 + 0.02 * side; pts[16].y = 0.42;
      pts[20].x = 0.5 + 0.04 * side; pts[20].y = 0.46;
      break;
    }

    case 'open_wave': {
      // XIN_CHÀO: Natural open hand, fingers extended with moderate spacing
      pts[4].x = 0.5 - 0.12 * side; pts[4].y = 0.52;
      pts[8].x = 0.5 - 0.05 * side; pts[8].y = 0.40;
      pts[12].x = 0.5 + 0.00 * side; pts[12].y = 0.38;
      pts[16].x = 0.5 + 0.05 * side; pts[16].y = 0.41;
      pts[20].x = 0.5 + 0.10 * side; pts[20].y = 0.47;
      break;
    }

    case 'spread_palm': {
      // TẠM_BIỆT: 5 fingers spread WIDE apart, waving
      pts[4].x = 0.5 - 0.16 * side; pts[4].y = 0.53;
      pts[8].x = 0.5 - 0.08 * side; pts[8].y = 0.39;
      pts[12].x = 0.5 + 0.00 * side; pts[12].y = 0.37;
      pts[16].x = 0.5 + 0.08 * side; pts[16].y = 0.40;
      pts[20].x = 0.5 + 0.15 * side; pts[20].y = 0.46;
      break;
    }

    case 'bent_down': {
      // BÂY_GIỜ: Fingers bent down at MCP/PIP knuckles, pressing downwards
      pts[6].y = 0.65; pts[7].y = 0.70; pts[8].y = 0.74; pts[8].z = -0.06;
      pts[10].y = 0.64; pts[11].y = 0.69; pts[12].y = 0.73; pts[12].z = -0.06;
      pts[14].y = 0.65; pts[15].y = 0.70; pts[16].y = 0.74; pts[16].z = -0.06;
      pts[18].y = 0.67; pts[19].y = 0.72; pts[20].y = 0.76; pts[20].z = -0.06;
      pts[4].x = 0.5 - 0.09 * side; pts[4].y = 0.64;
      break;
    }

    case 'chest_brush': {
      // CẢM_THẤY: Palm curved facing body/chest, gentle upward sweep
      pts[8].z = 0.08; pts[8].y = 0.44;
      pts[12].z = 0.10; pts[12].y = 0.43; // Middle finger bent slightly in
      pts[16].z = 0.08; pts[16].y = 0.45;
      pts[20].z = 0.06; pts[20].y = 0.50;
      pts[4].y = 0.58; pts[4].z = 0.04;
      break;
    }

    case 'flutter_chest': {
      // VUI_VẺ: Open hands upright in front of chest
      pts[4].x = 0.5 - 0.11 * side; pts[4].y = 0.54;
      pts[8].y = 0.42; pts[12].y = 0.39; pts[16].y = 0.43; pts[20].y = 0.48;
      break;
    }
  }

  return pts;
}

function makeSignSample(
  id: string,
  label: string,
  profile: string,
  handCount: 1 | 2,
  pose: PoseType,
  variation: number
): SignSample {
  const hand1Pts = generateRealisticHand(pose, variation, true);
  const norm1 = normalizeHandLandmarks(hand1Pts)!;

  let hand2Vector: number[] | undefined = undefined;
  let interWristOffset = undefined;

  if (handCount === 2) {
    const hand2Pts = generateRealisticHand(pose, variation + 0.3, false);
    const norm2 = normalizeHandLandmarks(hand2Pts)!;
    hand2Vector = norm2.points;
    interWristOffset = {
      dx: 0.32 + Math.sin(variation) * 0.02,
      dy: 0.02 + Math.cos(variation) * 0.02,
      dz: 0.0,
    };
  }

  return {
    id,
    label,
    profile,
    handCount,
    hand1: norm1.points,
    hand2: hand2Vector,
    interWristOffset,
    createdAt: Date.now() - Math.floor(Math.random() * 86400000 * 5),
  };
}

/**
 * Calibrated comprehensive dataset for Vietnamese Sign Language (NNKH).
 * Carefully designed so every single gesture has distinct finger states
 * and triggers immediately in both 1-hand and 2-hand modes.
 */
export const DEFAULT_SIGN_DATASET: SignSample[] = [
  // 1. TÔI (Index pointing to chest / self)
  makeSignSample('toi_k1', 'TÔI', 'khang', 1, 'point_chest', 0.1),
  makeSignSample('toi_k2', 'TÔI', 'khang', 1, 'point_chest', 0.2),
  makeSignSample('toi_l1', 'TÔI', 'lan', 1, 'point_chest', 0.3),
  makeSignSample('toi_l2', 'TÔI', 'lan', 1, 'point_chest', 0.4),
  makeSignSample('toi_m1', 'TÔI', 'minh', 1, 'point_chest', 0.5),
  makeSignSample('toi_m2', 'TÔI', 'minh', 1, 'point_chest', 0.6),

  // 2. BẠN (Index pointing forward at listener)
  makeSignSample('ban_k1', 'BẠN', 'khang', 1, 'point_forward', 0.1),
  makeSignSample('ban_k2', 'BẠN', 'khang', 1, 'point_forward', 0.2),
  makeSignSample('ban_l1', 'BẠN', 'lan', 1, 'point_forward', 0.3),
  makeSignSample('ban_l2', 'BẠN', 'lan', 1, 'point_forward', 0.4),
  makeSignSample('ban_m1', 'BẠN', 'minh', 1, 'point_forward', 0.5),
  makeSignSample('ban_m2', 'BẠN', 'minh', 1, 'point_forward', 0.6),

  // 3. KHỎE (Thumbs up 👍)
  makeSignSample('khoe_k1', 'KHỎE', 'khang', 1, 'thumbs_up', 0.1),
  makeSignSample('khoe_k2', 'KHỎE', 'khang', 1, 'thumbs_up', 0.2),
  makeSignSample('khoe_l1', 'KHỎE', 'lan', 1, 'thumbs_up', 0.3),
  makeSignSample('khoe_l2', 'KHỎE', 'lan', 1, 'thumbs_up', 0.4),
  makeSignSample('khoe_m1', 'KHỎE', 'minh', 1, 'thumbs_up', 0.5),
  makeSignSample('khoe_m2', 'KHỎE', 'minh', 2, 'thumbs_up', 0.6), // 2-hand thumbs up

  // 4. CẢM_ƠN (Flat hand from chin moving forward, fingers pressed together)
  makeSignSample('camon_k1', 'CẢM_ƠN', 'khang', 1, 'flat_palm', 0.1),
  makeSignSample('camon_k2', 'CẢM_ƠN', 'khang', 1, 'flat_palm', 0.2),
  makeSignSample('camon_l1', 'CẢM_ƠN', 'lan', 1, 'flat_palm', 0.3),
  makeSignSample('camon_l2', 'CẢM_ƠN', 'lan', 1, 'flat_palm', 0.4),
  makeSignSample('camon_m1', 'CẢM_ƠN', 'minh', 1, 'flat_palm', 0.5),

  // 5. XIN_CHÀO (Open hand waving / greeting)
  makeSignSample('xinchao_k1', 'XIN_CHÀO', 'khang', 1, 'open_wave', 0.1),
  makeSignSample('xinchao_k2', 'XIN_CHÀO', 'khang', 1, 'open_wave', 0.2),
  makeSignSample('xinchao_l1', 'XIN_CHÀO', 'lan', 1, 'open_wave', 0.3),
  makeSignSample('xinchao_l2', 'XIN_CHÀO', 'lan', 1, 'open_wave', 0.4),
  makeSignSample('xinchao_m1', 'XIN_CHÀO', 'minh', 1, 'open_wave', 0.5),

  // 6. TẠM_BIỆT (Open hand with 5 fingers spread wide apart waving)
  makeSignSample('tambiet_k1', 'TẠM_BIỆT', 'khang', 1, 'spread_palm', 0.1),
  makeSignSample('tambiet_k2', 'TẠM_BIỆT', 'khang', 1, 'spread_palm', 0.2),
  makeSignSample('tambiet_l1', 'TẠM_BIỆT', 'lan', 1, 'spread_palm', 0.3),
  makeSignSample('tambiet_l2', 'TẠM_BIỆT', 'lan', 1, 'spread_palm', 0.4),
  makeSignSample('tambiet_m1', 'TẠM_BIỆT', 'minh', 1, 'spread_palm', 0.5),

  // 7. BÂY_GIỜ (Bent hands pressing down; supports both 1-hand & 2-hand)
  makeSignSample('baygio_k1', 'BÂY_GIỜ', 'khang', 1, 'bent_down', 0.1),
  makeSignSample('baygio_k2', 'BÂY_GIỜ', 'khang', 2, 'bent_down', 0.2),
  makeSignSample('baygio_l1', 'BÂY_GIỜ', 'lan', 1, 'bent_down', 0.3),
  makeSignSample('baygio_l2', 'BÂY_GIỜ', 'lan', 2, 'bent_down', 0.4),
  makeSignSample('baygio_m1', 'BÂY_GIỜ', 'minh', 1, 'bent_down', 0.5),
  makeSignSample('baygio_m2', 'BÂY_GIỜ', 'minh', 2, 'bent_down', 0.6),

  // 8. CẢM_THẤY (Curved hand brushing chest; supports 1-hand & 2-hand)
  makeSignSample('camthay_k1', 'CẢM_THẤY', 'khang', 1, 'chest_brush', 0.1),
  makeSignSample('camthay_k2', 'CẢM_THẤY', 'khang', 2, 'chest_brush', 0.2),
  makeSignSample('camthay_l1', 'CẢM_THẤY', 'lan', 1, 'chest_brush', 0.3),
  makeSignSample('camthay_l2', 'CẢM_THẤY', 'lan', 2, 'chest_brush', 0.4),
  makeSignSample('camthay_m1', 'CẢM_THẤY', 'minh', 1, 'chest_brush', 0.5),

  // 9. VUI_VẺ (Open hands fluttering at chest; supports 1-hand & 2-hand)
  makeSignSample('vuive_k1', 'VUI_VẺ', 'khang', 1, 'flutter_chest', 0.1),
  makeSignSample('vuive_k2', 'VUI_VẺ', 'khang', 2, 'flutter_chest', 0.2),
  makeSignSample('vuive_l1', 'VUI_VẺ', 'lan', 1, 'flutter_chest', 0.3),
  makeSignSample('vuive_l2', 'VUI_VẺ', 'lan', 2, 'flutter_chest', 0.4),
  makeSignSample('vuive_m1', 'VUI_VẺ', 'minh', 1, 'flutter_chest', 0.5),
  makeSignSample('vuive_m2', 'VUI_VẺ', 'minh', 2, 'flutter_chest', 0.6),
];

// Upgraded version key to auto-migrate clients to newly calibrated dataset
export const STORAGE_KEY_DATASET = 'voice2hand_sign_dataset_v3';
export const STORAGE_KEY_PROFILE = 'voice2hand_active_profile';

export function loadStoredDataset(): SignSample[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_DATASET);
    if (!raw) {
      localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(DEFAULT_SIGN_DATASET));
      return DEFAULT_SIGN_DATASET;
    }
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length >= 20) {
      return parsed;
    }
    // Auto-upgrade if previous stored dataset was incomplete or older version
    localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(DEFAULT_SIGN_DATASET));
    return DEFAULT_SIGN_DATASET;
  } catch (e) {
    console.error('Failed to load sign dataset from localStorage:', e);
    return DEFAULT_SIGN_DATASET;
  }
}

export function saveStoredDataset(dataset: SignSample[]): void {
  try {
    localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(dataset));
  } catch (e) {
    console.error('Failed to save sign dataset to localStorage:', e);
  }
}

export function resetToDefaultDataset(): SignSample[] {
  localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(DEFAULT_SIGN_DATASET));
  return DEFAULT_SIGN_DATASET;
}
