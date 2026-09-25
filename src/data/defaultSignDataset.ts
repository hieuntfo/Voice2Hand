import { SignSample } from '../types';
import { normalizeHandLandmarks } from '../utils/handNormalization';

/**
 * Generates an anatomically structured 21-point hand skeleton for canonical sign poses.
 * 0: Wrist
 * 1-4: Thumb
 * 5-8: Index
 * 9-12: Middle
 * 13-16: Ring
 * 17-20: Pinky
 */
type PoseType =
  | 'point_chest'     // TÔI: Index pointing inward/upward towards chest, others curled
  | 'point_forward'   // BẠN: Index pointing forward, others curled
  | 'thumbs_up'       // KHỎE: Thumb extended straight up, 4 fingers curled tightly
  | 'open_wave'       // XIN_CHÀO / TẠM_BIỆT: 5 fingers fully extended, slight spread
  | 'flat_palm_front' // CẢM_ƠN: Fingers aligned flat together, moving forward
  | 'curved_chest'    // CẢM_THẤY: Open fingers gently curved upwards
  | 'bent_two_hands'  // BÂY_GIỜ: Bent palms facing up/down
  | 'flutter_chest';  // VUI_VẺ: Open hands upright with spread fingers

function generateHandSkeleton(pose: PoseType, variation: number = 0, isRightHand: boolean = true) {
  const side = isRightHand ? 1 : -1;
  const v = (seed: number) => (Math.sin(seed + variation * 10) * 0.02);

  // Baseline 21 points starting from wrist (0, 0, 0)
  const pts = [
    { x: 0.5 + v(1), y: 0.8 + v(2), z: 0 }, // 0: Wrist

    // Thumb (1, 2, 3, 4)
    { x: 0.5 - 0.05 * side + v(3), y: 0.74, z: -0.02 },
    { x: 0.5 - 0.09 * side + v(4), y: 0.69, z: -0.03 },
    { x: 0.5 - 0.12 * side + v(5), y: 0.64, z: -0.04 },
    { x: 0.5 - 0.14 * side + v(6), y: 0.60, z: -0.05 },

    // Index (5, 6, 7, 8)
    { x: 0.5 - 0.04 * side + v(7), y: 0.67, z: -0.01 },
    { x: 0.5 - 0.05 * side + v(8), y: 0.59, z: -0.01 },
    { x: 0.5 - 0.06 * side + v(9), y: 0.53, z: -0.01 },
    { x: 0.5 - 0.07 * side + v(10), y: 0.47, z: -0.01 },

    // Middle (9, 10, 11, 12)
    { x: 0.5 + 0.00 * side + v(11), y: 0.66, z: 0 },
    { x: 0.5 + 0.00 * side + v(12), y: 0.57, z: 0 },
    { x: 0.5 + 0.00 * side + v(13), y: 0.50, z: 0 },
    { x: 0.5 + 0.00 * side + v(14), y: 0.44, z: 0 },

    // Ring (13, 14, 15, 16)
    { x: 0.5 + 0.04 * side + v(15), y: 0.67, z: 0.01 },
    { x: 0.5 + 0.05 * side + v(16), y: 0.59, z: 0.01 },
    { x: 0.5 + 0.06 * side + v(17), y: 0.53, z: 0.01 },
    { x: 0.5 + 0.07 * side + v(18), y: 0.47, z: 0.01 },

    // Pinky (17, 18, 19, 20)
    { x: 0.5 + 0.08 * side + v(19), y: 0.69, z: 0.02 },
    { x: 0.5 + 0.10 * side + v(20), y: 0.63, z: 0.02 },
    { x: 0.5 + 0.11 * side + v(21), y: 0.58, z: 0.02 },
    { x: 0.5 + 0.12 * side + v(22), y: 0.53, z: 0.02 },
  ];

  // Adjust finger tip and joint positions based on pose type
  switch (pose) {
    case 'point_chest': {
      // Index points back / up towards chest, thumb curled, middle/ring/pinky curled in
      pts[7].y = 0.65; pts[8].y = 0.68; pts[8].z = 0.12; // Index curled backward toward chest
      pts[8].x = 0.5 - 0.02 * side;
      // Curl other fingers in
      [10, 11, 12].forEach(i => { pts[i].y = 0.72; pts[i].z = 0.06; });
      [14, 15, 16].forEach(i => { pts[i].y = 0.73; pts[i].z = 0.06; });
      [18, 19, 20].forEach(i => { pts[i].y = 0.75; pts[i].z = 0.05; });
      pts[4].x = 0.5 - 0.03 * side; pts[4].y = 0.70; // Thumb holding fingers
      break;
    }

    case 'point_forward': {
      // Index points straight forward (deep z or straight up in 2D)
      pts[7].y = 0.50; pts[8].y = 0.42; pts[8].z = -0.15;
      [10, 11, 12].forEach(i => { pts[i].y = 0.71; pts[i].z = 0.05; });
      [14, 15, 16].forEach(i => { pts[i].y = 0.72; pts[i].z = 0.05; });
      [18, 19, 20].forEach(i => { pts[i].y = 0.74; pts[i].z = 0.05; });
      pts[4].x = 0.5 - 0.02 * side; pts[4].y = 0.68;
      break;
    }

    case 'thumbs_up': {
      // Thumb extended straight up
      pts[2].y = 0.65; pts[3].y = 0.57; pts[4].y = 0.48;
      pts[4].x = 0.5 - 0.06 * side;
      // All other fingers curled tightly into palm
      [6, 7, 8].forEach(i => { pts[i].y = 0.70; pts[i].z = 0.05; });
      [10, 11, 12].forEach(i => { pts[i].y = 0.71; pts[i].z = 0.05; });
      [14, 15, 16].forEach(i => { pts[i].y = 0.72; pts[i].z = 0.05; });
      [18, 19, 20].forEach(i => { pts[i].y = 0.74; pts[i].z = 0.05; });
      break;
    }

    case 'open_wave': {
      // 5 fingers spread out, waving posture
      pts[4].x -= 0.04 * side;
      pts[8].x -= 0.02 * side;
      pts[20].x += 0.03 * side;
      break;
    }

    case 'flat_palm_front': {
      // Fingers together, flat palm
      pts[4].x = 0.5 - 0.06 * side; pts[4].y = 0.62;
      pts[8].x = 0.5 - 0.03 * side; pts[8].y = 0.46;
      pts[12].x = 0.5 + 0.00 * side; pts[12].y = 0.44;
      pts[16].x = 0.5 + 0.03 * side; pts[16].y = 0.46;
      pts[20].x = 0.5 + 0.06 * side; pts[20].y = 0.50;
      break;
    }

    case 'curved_chest': {
      // Gentle curve of all fingers facing chest
      [8, 12, 16, 20].forEach(i => { pts[i].z = 0.08; pts[i].y += 0.04; });
      break;
    }

    case 'bent_two_hands': {
      // Two hands held in front, fingers bent downward at knuckles
      [7, 8, 11, 12, 15, 16, 19, 20].forEach(i => { pts[i].y += 0.08; pts[i].z = -0.06; });
      break;
    }

    case 'flutter_chest': {
      // Two hands upright in front of chest, open fingers slightly curved upward
      pts[4].x -= 0.03 * side;
      pts[8].y = 0.44; pts[12].y = 0.42; pts[16].y = 0.45; pts[20].y = 0.49;
      break;
    }
  }

  return pts;
}

function makeSample(
  id: string,
  label: string,
  profile: string,
  handCount: 1 | 2,
  pose: PoseType,
  variation: number,
  notes?: string
): SignSample {
  const hand1Pts = generateHandSkeleton(pose, variation, true);
  const norm1 = normalizeHandLandmarks(hand1Pts)!;

  let hand2Vector: number[] | undefined = undefined;
  let interWristOffset = undefined;

  if (handCount === 2) {
    const hand2Pts = generateHandSkeleton(pose, variation + 0.5, false);
    const norm2 = normalizeHandLandmarks(hand2Pts)!;
    hand2Vector = norm2.points;
    interWristOffset = {
      dx: 0.35 + Math.sin(variation) * 0.02,
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
    createdAt: Date.now() - Math.floor(Math.random() * 86400000 * 7),
    notes,
  };
}

/**
 * Pre-seeded Sign Language Dataset for Vietnamese Sign Language (NNKH)
 * Includes the 9 standard symbols:
 * TÔI, BÂY_GIỜ, CẢM_THẤY, VUI_VẺ, XIN_CHÀO, CẢM_ƠN, TẠM_BIỆT, BẠN, KHỎE
 * Multi-profile: "khang", "lan", "minh" to demonstrate Profile vs Global Model.
 */
export const DEFAULT_SIGN_DATASET: SignSample[] = [
  // 1. TÔI (1 hand) - Pointing at chest
  makeSample('s_toi_1', 'TÔI', 'khang', 1, 'point_chest', 0.1, 'Chỉ tay vào ngực'),
  makeSample('s_toi_2', 'TÔI', 'khang', 1, 'point_chest', 0.2),
  makeSample('s_toi_3', 'TÔI', 'khang', 1, 'point_chest', 0.3),
  makeSample('s_toi_4', 'TÔI', 'lan', 1, 'point_chest', 0.4),
  makeSample('s_toi_5', 'TÔI', 'lan', 1, 'point_chest', 0.5),
  makeSample('s_toi_6', 'TÔI', 'minh', 1, 'point_chest', 0.6),

  // 2. BÂY_GIỜ (2 hands) - Bent hands pressing downward in front
  makeSample('s_baygio_1', 'BÂY_GIỜ', 'khang', 2, 'bent_two_hands', 0.1, 'Hai tay gập nhẹ nhấn xuống'),
  makeSample('s_baygio_2', 'BÂY_GIỜ', 'khang', 2, 'bent_two_hands', 0.2),
  makeSample('s_baygio_3', 'BÂY_GIỜ', 'khang', 2, 'bent_two_hands', 0.3),
  makeSample('s_baygio_4', 'BÂY_GIỜ', 'lan', 2, 'bent_two_hands', 0.4),
  makeSample('s_baygio_5', 'BÂY_GIỜ', 'minh', 2, 'bent_two_hands', 0.5),

  // 3. CẢM_THẤY (1 hand / 2 hands) - Hand moving gently up chest
  makeSample('s_camthay_1', 'CẢM_THẤY', 'khang', 1, 'curved_chest', 0.1, 'Lòng bàn tay hướng ngực vuốt lên'),
  makeSample('s_camthay_2', 'CẢM_THẤY', 'khang', 1, 'curved_chest', 0.2),
  makeSample('s_camthay_3', 'CẢM_THẤY', 'lan', 1, 'curved_chest', 0.3),
  makeSample('s_camthay_4', 'CẢM_THẤY', 'lan', 1, 'curved_chest', 0.4),
  makeSample('s_camthay_5', 'CẢM_THẤY', 'minh', 1, 'curved_chest', 0.5),

  // 4. VUI_VẺ (2 hands) - Fluttering upright near chest
  makeSample('s_vuive_1', 'VUI_VẺ', 'khang', 2, 'flutter_chest', 0.1, 'Hai bàn tay mở vỗ nhẹ hướng ngực'),
  makeSample('s_vuive_2', 'VUI_VẺ', 'khang', 2, 'flutter_chest', 0.2),
  makeSample('s_vuive_3', 'VUI_VẺ', 'khang', 2, 'flutter_chest', 0.3),
  makeSample('s_vuive_4', 'VUI_VẺ', 'lan', 2, 'flutter_chest', 0.4),
  makeSample('s_vuive_5', 'VUI_VẺ', 'minh', 2, 'flutter_chest', 0.5),

  // 5. XIN_CHÀO (1 hand) - Waving / saluting
  makeSample('s_xinchao_1', 'XIN_CHÀO', 'khang', 1, 'open_wave', 0.1, 'Vẫy tay chào'),
  makeSample('s_xinchao_2', 'XIN_CHÀO', 'khang', 1, 'open_wave', 0.2),
  makeSample('s_xinchao_3', 'XIN_CHÀO', 'lan', 1, 'open_wave', 0.3),
  makeSample('s_xinchao_4', 'XIN_CHÀO', 'minh', 1, 'open_wave', 0.4),
  makeSample('s_xinchao_5', 'XIN_CHÀO', 'minh', 1, 'open_wave', 0.5),

  // 6. CẢM_ƠN (1 hand) - Flat hand from chin moving forward
  makeSample('s_camon_1', 'CẢM_ƠN', 'khang', 1, 'flat_palm_front', 0.1, 'Bàn tay từ cằm đưa ra trước'),
  makeSample('s_camon_2', 'CẢM_ƠN', 'khang', 1, 'flat_palm_front', 0.2),
  makeSample('s_camon_3', 'CẢM_ƠN', 'lan', 1, 'flat_palm_front', 0.3),
  makeSample('s_camon_4', 'CẢM_ƠN', 'lan', 1, 'flat_palm_front', 0.4),
  makeSample('s_camon_5', 'CẢM_ƠN', 'minh', 1, 'flat_palm_front', 0.5),

  // 7. TẠM_BIỆT (1 hand) - Open palm waving
  makeSample('s_tambiet_1', 'TẠM_BIỆT', 'khang', 1, 'open_wave', 0.6, 'Vẫy tay tạm biệt'),
  makeSample('s_tambiet_2', 'TẠM_BIỆT', 'khang', 1, 'open_wave', 0.7),
  makeSample('s_tambiet_3', 'TẠM_BIỆT', 'lan', 1, 'open_wave', 0.8),
  makeSample('s_tambiet_4', 'TẠM_BIỆT', 'minh', 1, 'open_wave', 0.9),
  makeSample('s_tambiet_5', 'TẠM_BIỆT', 'minh', 1, 'open_wave', 1.0),

  // 8. BẠN (1 hand) - Index finger pointing forward
  makeSample('s_ban_1', 'BẠN', 'khang', 1, 'point_forward', 0.1, 'Chỉ tay về phía người nghe'),
  makeSample('s_ban_2', 'BẠN', 'khang', 1, 'point_forward', 0.2),
  makeSample('s_ban_3', 'BẠN', 'lan', 1, 'point_forward', 0.3),
  makeSample('s_ban_4', 'BẠN', 'minh', 1, 'point_forward', 0.4),
  makeSample('s_ban_5', 'BẠN', 'minh', 1, 'point_forward', 0.5),

  // 9. KHỎE (1 hand) - Thumbs up
  makeSample('s_khoe_1', 'KHỎE', 'khang', 1, 'thumbs_up', 0.1, 'Ngón cái giơ lên dứt khoát'),
  makeSample('s_khoe_2', 'KHỎE', 'khang', 1, 'thumbs_up', 0.2),
  makeSample('s_khoe_3', 'KHỎE', 'lan', 1, 'thumbs_up', 0.3),
  makeSample('s_khoe_4', 'KHỎE', 'lan', 1, 'thumbs_up', 0.4),
  makeSample('s_khoe_5', 'KHỎE', 'minh', 1, 'thumbs_up', 0.5),
];

export const STORAGE_KEY_DATASET = 'voice2hand_sign_dataset_v1';
export const STORAGE_KEY_PROFILE = 'voice2hand_active_profile';

export function loadStoredDataset(): SignSample[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_DATASET);
    if (!raw) {
      localStorage.setItem(STORAGE_KEY_DATASET, JSON.stringify(DEFAULT_SIGN_DATASET));
      return DEFAULT_SIGN_DATASET;
    }
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) {
      return parsed;
    }
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
