import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { Landmark } from '../types';
import { HAND_CONNECTIONS } from './handNormalization';

let handLandmarkerInstance: HandLandmarker | null = null;
let isInitializing = false;
let initPromise: Promise<HandLandmarker | null> | null = null;

const WASM_CDN_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_ASSET_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

/**
 * Initializes and caches the MediaPipe HandLandmarker.
 */
export async function getHandLandmarker(): Promise<HandLandmarker | null> {
  if (handLandmarkerInstance) {
    return handLandmarkerInstance;
  }

  if (initPromise) {
    return initPromise;
  }

  isInitializing = true;
  initPromise = (async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_CDN_URL);

      try {
        // Attempt GPU first for high fps
        handLandmarkerInstance = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_URL,
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      } catch (gpuError) {
        console.warn('GPU delegate failed for MediaPipe, falling back to CPU:', gpuError);
        handLandmarkerInstance = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_URL,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
      }

      isInitializing = false;
      return handLandmarkerInstance;
    } catch (err) {
      console.error('Failed to initialize MediaPipe HandLandmarker:', err);
      isInitializing = false;
      initPromise = null;
      return null;
    }
  })();

  return initPromise;
}

/**
 * Draws 2D green wireframe landmarks and nodes over video canvas.
 */
export function drawLandmarksOnCanvas(
  ctx: CanvasRenderingContext2D,
  handsLandmarks: Landmark[][],
  canvasWidth: number,
  canvasHeight: number,
  isMirrored: boolean = true
): void {
  ctx.save();
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);

  if (isMirrored) {
    ctx.translate(canvasWidth, 0);
    ctx.scale(-1, 1);
  }

  for (let hIdx = 0; hIdx < handsLandmarks.length; hIdx++) {
    const landmarks = handsLandmarks[hIdx];
    if (!landmarks || landmarks.length !== 21) continue;

    // Draw skeletal connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#22c55e'; // Bright vibrant green
    ctx.shadowColor = 'rgba(34, 197, 94, 0.6)';
    ctx.shadowBlur = 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];

      const x1 = p1.x * canvasWidth;
      const y1 = p1.y * canvasHeight;
      const x2 = p2.x * canvasWidth;
      const y2 = p2.y * canvasHeight;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw joint nodes
    const tipIndices = new Set([4, 8, 12, 16, 20]);

    for (let i = 0; i < landmarks.length; i++) {
      const p = landmarks[i];
      const x = p.x * canvasWidth;
      const y = p.y * canvasHeight;

      const isTip = tipIndices.has(i);
      const isWrist = i === 0;

      ctx.beginPath();
      if (isTip) {
        // Glowing fingertip
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#4ade80';
        ctx.shadowColor = '#4ade80';
        ctx.shadowBlur = 10;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      } else if (isWrist) {
        // Wrist root
        ctx.arc(x, y, 7, 0, 2 * Math.PI);
        ctx.fillStyle = '#10b981';
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 8;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      } else {
        // Standard joint
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#86efac';
        ctx.shadowColor = 'rgba(74, 222, 128, 0.5)';
        ctx.shadowBlur = 4;
        ctx.fill();
      }
    }
  }

  ctx.restore();
}
