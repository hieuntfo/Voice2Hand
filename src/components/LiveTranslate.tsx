import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Volume2,
  Trash2,
  Sparkles,
  Camera,
  AlertTriangle,
  CheckCircle2,
  Users,
  Info,
  Layers,
  Zap,
  Sliders,
  X,
} from 'lucide-react';
import { SignSample, RecognitionResult, Landmark } from '../types';
import {
  getHandLandmarker,
  drawLandmarksOnCanvas,
} from '../utils/mediaPipeService';
import {
  normalizeHandLandmarks,
} from '../utils/handNormalization';
import {
  classifyHandGesture,
  FrameInputData,
} from '../utils/knnClassifier';
import {
  SentenceAggregator,
  AggregatorConfig,
  DEFAULT_AGGREGATOR_CONFIG,
} from '../utils/sentenceBuilder';
import { speakVietnamese } from '../utils/speech';

interface LiveTranslateProps {
  dataset: SignSample[];
  activeProfile: string;
  onProfileChange: (profile: string) => void;
  availableProfiles: string[];
}

type SensitivityPreset = 'fast' | 'balanced' | 'precise';

const PRESETS: Record<SensitivityPreset, AggregatorConfig> = {
  fast: {
    stabilityFrames: 3,         // ~90-120ms lock time
    cooldownMs: 320,            // Fast 320ms transition
    restFramesRequired: 2,
    confidenceThreshold: 0.48,  // Very responsive
  },
  balanced: {
    stabilityFrames: 4,         // ~140-160ms lock time
    cooldownMs: 420,            // 420ms transition
    restFramesRequired: 2,
    confidenceThreshold: 0.52,  // Default balanced
  },
  precise: {
    stabilityFrames: 6,         // ~220ms lock time
    cooldownMs: 650,            // 650ms transition
    restFramesRequired: 3,
    confidenceThreshold: 0.60,  // Higher threshold
  },
};

/**
 * Plays a short, soft audio cue when a gesture is recognized and committed.
 */
function playCommitSound() {
  try {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(659.25, ctx.currentTime); // E5
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.08); // A5
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.12);
  } catch {
    // AudioContext not allowed or not supported; ignore silently
  }
}

export const LiveTranslate: React.FC<LiveTranslateProps> = ({
  dataset,
  activeProfile,
  onProfileChange,
  availableProfiles,
}) => {
  // Video & Canvas elements
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const landmarkerInstanceRef = useRef<any>(null);
  const lastProcessTimeRef = useRef<number>(0);

  // Camera & Model state
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);

  // Sensitivity Settings State
  const [currentPreset, setCurrentPreset] = useState<SensitivityPreset>('balanced');
  const [sensitivityConfig, setSensitivityConfig] = useState<AggregatorConfig>(PRESETS.balanced);
  const [showSensitivityPanel, setShowSensitivityPanel] = useState<boolean>(false);

  // Recognition state
  const [recognitionResult, setRecognitionResult] = useState<RecognitionResult>({
    label: '',
    confidence: 0,
    handCount: 0,
    isConfident: false,
    statusText: 'Đang chờ tay...',
    topMatches: [],
  });

  // Sentence state
  const [sentenceTokens, setSentenceTokens] = useState<string[]>([]);
  const [recentCommitBadge, setRecentCommitBadge] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [gestureProgress, setGestureProgress] = useState<number>(0);
  const [isCoolingDown, setIsCoolingDown] = useState<boolean>(false);

  // Optional: Gemini Smoothed Sentence
  const [isSmootherEnabled, setIsSmootherEnabled] = useState<boolean>(false);
  const [smoothedText, setSmoothedText] = useState<string | null>(null);
  const [isSmoothingLoading, setIsSmoothingLoading] = useState<boolean>(false);
  const [smoothingError, setSmoothingError] = useState<string | null>(null);

  // Temporal aggregator ref
  const aggregatorRef = useRef<SentenceAggregator>(
    new SentenceAggregator(PRESETS.balanced)
  );

  // Distinct labels in current pool
  const activePool =
    activeProfile === 'global'
      ? dataset
      : dataset.filter((s) => s.profile.toLowerCase() === activeProfile.toLowerCase());
  const distinctLabels = Array.from(new Set(activePool.map((s) => s.label)));

  // Show toast notification
  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage((prev) => (prev === msg ? null : prev));
    }, 2200);
  };

  // Change sensitivity preset
  const handleSelectPreset = (preset: SensitivityPreset) => {
    setCurrentPreset(preset);
    const newConfig = PRESETS[preset];
    setSensitivityConfig(newConfig);
    aggregatorRef.current.updateConfig(newConfig);
    showToast(`Độ nhạy: ${preset === 'fast' ? 'Siêu nhanh' : preset === 'balanced' ? 'Tiêu chuẩn' : 'Chính xác'}`);
  };

  // Start Camera
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError('Trình duyệt này không hỗ trợ truy cập camera.');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setCameraActive(true);
        };
      }
    } catch (err: any) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraError(
          'Chưa cấp quyền camera. Vui lòng cho phép truy cập camera trong cài đặt trình duyệt để tiếp tục.'
        );
      } else if (err.name === 'NotFoundError') {
        setCameraError('Không tìm thấy thiết bị camera trên máy tính của bạn.');
      } else {
        setCameraError(`Không thể khởi động camera: ${err.message || 'Lỗi không xác định'}`);
      }
      setCameraActive(false);
    }
  }, []);

  // Stop Camera
  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  // Initialize MediaPipe model
  useEffect(() => {
    let mounted = true;
    setModelLoading(true);

    getHandLandmarker()
      .then((landmarker) => {
        if (!mounted) return;
        if (landmarker) {
          landmarkerInstanceRef.current = landmarker;
          setModelLoading(false);
          startCamera();
        } else {
          setCameraError('Không thể tải mô hình MediaPipe Tasks Vision.');
          setModelLoading(false);
        }
      })
      .catch((err) => {
        if (!mounted) return;
        console.error('Init landmarker error:', err);
        setCameraError('Lỗi khởi tạo mô hình nhận diện tay.');
        setModelLoading(false);
      });

    return () => {
      mounted = false;
      stopCamera();
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [startCamera, stopCamera]);

  // Main Detection Loop (Optimized for smooth, low-latency execution)
  useEffect(() => {
    let isRunning = true;
    const TARGET_INTERVAL_MS = 32; // ~30 FPS: optimal for low-latency & no thermal throttling on mobile

    function processVideoFrame(now: number) {
      if (!isRunning) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerInstanceRef.current;

      // Throttle detection to ~30 FPS to avoid microtask lag and frame drops
      const elapsed = now - lastProcessTimeRef.current;

      if (
        landmarker &&
        video &&
        canvas &&
        video.readyState >= 2 &&
        !video.paused &&
        !video.ended &&
        elapsed >= TARGET_INTERVAL_MS
      ) {
        lastProcessTimeRef.current = now;

        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
        }

        const ctx = canvas.getContext('2d');
        const timestamp = performance.now();

        try {
          const results = landmarker.detectForVideo(video, timestamp);
          const handsLandmarks = (results.landmarks as Landmark[][]) || [];

          // Draw overlay landmarks on canvas
          if (ctx) {
            drawLandmarksOnCanvas(ctx, handsLandmarks, canvas.width, canvas.height, true);
          }

          // Extract normalized feature vectors
          let frameInput: FrameInputData = { handCount: 0 };

          if (handsLandmarks.length === 1) {
            const norm1 = normalizeHandLandmarks(handsLandmarks[0]);
            if (norm1) {
              frameInput = {
                handCount: 1,
                hand1Points: norm1.points,
                fingerStates1: norm1.fingerStates,
              };
            }
          } else if (handsLandmarks.length >= 2) {
            // Up to 2 hands
            const norm1 = normalizeHandLandmarks(handsLandmarks[0]);
            const norm2 = normalizeHandLandmarks(handsLandmarks[1]);
            if (norm1 && norm2) {
              const avgScale = (norm1.scale + norm2.scale) / 2 || 1;
              frameInput = {
                handCount: 2,
                hand1Points: norm1.points,
                hand2Points: norm2.points,
                fingerStates1: norm1.fingerStates,
                fingerStates2: norm2.fingerStates,
                interWristOffset: {
                  dx: (norm2.wrist.x - norm1.wrist.x) / avgScale,
                  dy: (norm2.wrist.y - norm1.wrist.y) / avgScale,
                  dz: ((norm2.wrist.z ?? 0) - (norm1.wrist.z ?? 0)) / avgScale,
                },
              };
            }
          }

          // Classify with k-NN using current sensitivity threshold
          const recognition = classifyHandGesture(
            frameInput,
            dataset,
            activeProfile,
            3,
            sensitivityConfig.confidenceThreshold
          );
          setRecognitionResult(recognition);

          // Temporal Segmentation & Sentence building
          const { committedToken, progress, isCoolingDown: cooldown } =
            aggregatorRef.current.processFrame(recognition);

          setGestureProgress(progress);
          setIsCoolingDown(cooldown);

          if (committedToken) {
            playCommitSound();
            setRecentCommitBadge(committedToken);
            setTimeout(() => setRecentCommitBadge(null), 1800);

            setSentenceTokens((prev) => [...prev, committedToken]);
          }
        } catch (detErr) {
          console.error('Detection frame error:', detErr);
        }
      }

      if (isRunning) {
        animFrameIdRef.current = requestAnimationFrame(processVideoFrame);
      }
    }

    animFrameIdRef.current = requestAnimationFrame(processVideoFrame);

    return () => {
      isRunning = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [dataset, activeProfile, sensitivityConfig]);

  // Clear Sentence
  const handleClearSentence = () => {
    setSentenceTokens([]);
    setSmoothedText(null);
    setSmoothingError(null);
    aggregatorRef.current.reset();
    showToast('Đã xoá câu');
  };

  // Remove a single token
  const handleRemoveToken = (indexToRemove: number) => {
    setSentenceTokens((prev) => prev.filter((_, idx) => idx !== indexToRemove));
  };

  // Text to Speech
  const handleSpeak = () => {
    const textToSpeak = smoothedText || sentenceTokens.join(' ');
    if (!textToSpeak.trim()) return;
    speakVietnamese(textToSpeak);
  };

  // Optional: Gemini Sentence Smoothing
  const handleSmoothSentence = async () => {
    if (sentenceTokens.length === 0) return;
    setIsSmoothingLoading(true);
    setSmoothingError(null);

    try {
      const response = await fetch('/api/smooth-sentence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tokens: sentenceTokens,
          rawText: sentenceTokens.join(' '),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.smoothedText) {
        setSmoothedText(data.smoothedText);
      }
    } catch (err: any) {
      console.error('Error smoothing sentence:', err);
      const fallback = sentenceTokens
        .map((t) => t.replace(/_/g, ' ').toLowerCase())
        .join(' ');
      const clean = fallback.charAt(0).toUpperCase() + fallback.slice(1) + '.';
      setSmoothedText(clean);
      setSmoothingError('Đã áp dụng định dạng cơ bản.');
    } finally {
      setIsSmoothingLoading(false);
    }
  };

  // Quick gesture simulator
  const handleSimulateGesture = (label: string) => {
    const simulatedRecognition: RecognitionResult = {
      label,
      confidence: 0.95,
      handCount: ['BÂY_GIỜ', 'VUI_VẺ'].includes(label) ? 2 : 1,
      isConfident: true,
      statusText: `✓ ${label} · confidence 95%`,
      topMatches: [{ label, score: 0.95 }],
    };

    setRecognitionResult(simulatedRecognition);

    for (let i = 0; i < sensitivityConfig.stabilityFrames; i++) {
      const { committedToken } = aggregatorRef.current.processFrame(simulatedRecognition);
      if (committedToken) {
        playCommitSound();
        setRecentCommitBadge(committedToken);
        setTimeout(() => setRecentCommitBadge(null), 1800);
        setSentenceTokens((prev) => [...prev, committedToken]);
        break;
      }
    }
  };

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto px-3 sm:px-4 py-2 sm:py-4 space-y-3 sm:space-y-4">
      {/* Toast Notification (Floating unobtrusively at top center) */}
      {toastMessage && (
        <div className="fixed top-14 left-1/2 -translate-x-1/2 z-50 bg-neutral-900/95 border border-emerald-500/50 text-emerald-400 px-4 py-2 rounded-full shadow-2xl text-xs font-mono flex items-center gap-2 animate-in fade-in slide-in-from-top-2 duration-150">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Profile & Speed Selector Bar */}
      <div className="w-full flex flex-col sm:flex-row items-center justify-between bg-neutral-900/90 border border-neutral-800 rounded-xl px-3 sm:px-4 py-2 gap-2 text-xs">
        {/* Profile Selector */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Users className="w-4 h-4 text-neutral-400 shrink-0" />
          <span className="text-neutral-400 font-medium shrink-0">Mẫu:</span>
          <select
            value={activeProfile}
            onChange={(e) => onProfileChange(e.target.value)}
            className="flex-1 sm:flex-initial bg-neutral-950 border border-neutral-700 text-neutral-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-emerald-500 font-medium cursor-pointer"
          >
            <option value="global">🌐 Global Model (Toàn bộ)</option>
            {availableProfiles.map((p) => (
              <option key={p} value={p}>
                👤 Profile: {p}
              </option>
            ))}
          </select>
        </div>

        {/* Speed & Sensitivity Presets Bar */}
        <div className="flex items-center gap-1.5 w-full sm:w-auto justify-between sm:justify-end">
          <div className="flex items-center gap-1 bg-neutral-950 p-0.5 rounded-lg border border-neutral-800">
            <button
              onClick={() => handleSelectPreset('fast')}
              className={`px-2 py-1 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                currentPreset === 'fast'
                  ? 'bg-emerald-500 text-neutral-950 font-bold shadow'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              🚀 Nhanh
            </button>
            <button
              onClick={() => handleSelectPreset('balanced')}
              className={`px-2 py-1 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                currentPreset === 'balanced'
                  ? 'bg-emerald-500 text-neutral-950 font-bold shadow'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              ⚡ Chuẩn
            </button>
            <button
              onClick={() => handleSelectPreset('precise')}
              className={`px-2 py-1 rounded text-[11px] font-semibold transition-all cursor-pointer ${
                currentPreset === 'precise'
                  ? 'bg-emerald-500 text-neutral-950 font-bold shadow'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              🎯 Kỹ
            </button>
          </div>

          <button
            onClick={() => setShowSensitivityPanel(!showSensitivityPanel)}
            title="Tùy chỉnh thông số độ nhạy"
            className={`p-1.5 rounded-lg border transition-colors cursor-pointer ${
              showSensitivityPanel
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                : 'bg-neutral-950 text-neutral-400 border-neutral-800 hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Optional Expanded Sensitivity Slider Panel */}
      {showSensitivityPanel && (
        <div className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-3.5 space-y-3 text-xs animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
            <span className="font-bold text-white flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-emerald-400" />
              Tùy chỉnh thông số nhận diện & tốc độ
            </span>
            <button
              onClick={() => setShowSensitivityPanel(false)}
              className="text-neutral-500 hover:text-white p-0.5"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* 1. Frames required */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-neutral-400">
                <span>Số khung hình chốt:</span>
                <strong className="text-emerald-400">{sensitivityConfig.stabilityFrames} frames (~{Math.round(sensitivityConfig.stabilityFrames * 33)}ms)</strong>
              </div>
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={sensitivityConfig.stabilityFrames}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  const updated = { ...sensitivityConfig, stabilityFrames: val };
                  setSensitivityConfig(updated);
                  aggregatorRef.current.updateConfig(updated);
                }}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <span className="text-[10px] text-neutral-500">Giảm để nhận diện nhanh hơn</span>
            </div>

            {/* 2. Cooldown */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-neutral-400">
                <span>Thời gian hồi chiêu:</span>
                <strong className="text-emerald-400">{sensitivityConfig.cooldownMs}ms</strong>
              </div>
              <input
                type="range"
                min="250"
                max="900"
                step="50"
                value={sensitivityConfig.cooldownMs}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  const updated = { ...sensitivityConfig, cooldownMs: val };
                  setSensitivityConfig(updated);
                  aggregatorRef.current.updateConfig(updated);
                }}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <span className="text-[10px] text-neutral-500">Nghỉ giữa 2 ký hiệu liên tiếp</span>
            </div>

            {/* 3. Threshold */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-neutral-400">
                <span>Ngưỡng tin cậy (Confidence):</span>
                <strong className="text-emerald-400">{Math.round(sensitivityConfig.confidenceThreshold * 100)}%</strong>
              </div>
              <input
                type="range"
                min="0.40"
                max="0.75"
                step="0.02"
                value={sensitivityConfig.confidenceThreshold}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  const updated = { ...sensitivityConfig, confidenceThreshold: val };
                  setSensitivityConfig(updated);
                  aggregatorRef.current.updateConfig(updated);
                }}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <span className="text-[10px] text-neutral-500">Giảm nếu cử chỉ tay hay bị "Không chắc chắn"</span>
            </div>
          </div>
        </div>
      )}

      {/* Centered Camera Container */}
      <div className="w-full max-w-2xl flex flex-col items-center">
        <div className="relative w-full aspect-[4/3] bg-neutral-950 rounded-2xl border-2 border-neutral-800 hover:border-emerald-500/40 transition-colors overflow-hidden shadow-2xl flex items-center justify-center">
          {/* Label "TRANSLATION" at top-left */}
          <div className="absolute top-3 left-3 z-20 flex items-center gap-1.5 bg-neutral-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-neutral-800 text-[11px] font-mono tracking-wider text-emerald-400 font-bold uppercase select-none">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
            TRANSLATION
          </div>

          {/* Hand Count Indicator at top-right */}
          <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-neutral-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-neutral-800 text-[11px] font-mono text-neutral-400 select-none">
            <Layers className="w-3 h-3 text-neutral-400" />
            <span>
              {recognitionResult.handCount === 0
                ? '0 tay'
                : `${recognitionResult.handCount} tay`}
            </span>
          </div>

          {/* Quick Commit Badge on screen */}
          {recentCommitBadge && (
            <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
              <div className="bg-emerald-500/90 text-neutral-950 px-5 py-2 rounded-2xl text-xl sm:text-2xl font-black font-mono tracking-wider shadow-[0_0_30px_#22c55e] animate-out zoom-out-90 duration-500">
                ✓ {recentCommitBadge}
              </div>
            </div>
          )}

          {/* Video Feed (mirrored for natural interaction) */}
          <video
            ref={videoRef}
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-cover -scale-x-100"
          />

          {/* Landmark Overlay Canvas */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none"
          />

          {/* Camera Loading or Error State */}
          {(!cameraActive || modelLoading) && (
            <div className="absolute inset-0 z-30 bg-neutral-950/95 flex flex-col items-center justify-center p-6 text-center space-y-3">
              {modelLoading ? (
                <>
                  <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  <p className="text-sm font-medium text-neutral-300">
                    Đang tải mô hình MediaPipe Vision...
                  </p>
                  <p className="text-xs text-neutral-500">Khởi tạo HandLandmarker 2 tay</p>
                </>
              ) : cameraError ? (
                <div className="space-y-3 max-w-sm">
                  <div className="w-10 h-10 rounded-full bg-amber-500/10 text-amber-400 flex items-center justify-center mx-auto border border-amber-500/20">
                    <AlertTriangle className="w-5 h-5" />
                  </div>
                  <p className="text-sm text-neutral-200 font-medium">{cameraError}</p>
                  <button
                    onClick={startCamera}
                    className="px-4 py-2 bg-emerald-500 text-neutral-950 font-bold rounded-lg text-xs hover:bg-emerald-400 transition-colors shadow cursor-pointer"
                  >
                    Thử lại quyền camera
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <Camera className="w-10 h-10 text-neutral-600 mx-auto" />
                  <p className="text-sm text-neutral-400">Camera đang tạm dừng</p>
                  <button
                    onClick={startCamera}
                    className="px-4 py-2 bg-emerald-500 text-neutral-950 font-bold rounded-lg text-xs hover:bg-emerald-400 transition-colors cursor-pointer"
                  >
                    Bật Camera
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Temporal Confirmation Progress Bar */}
          {gestureProgress > 0 && gestureProgress < 1 && (
            <div className="absolute bottom-0 left-0 right-0 h-2 bg-neutral-900/80 z-20 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-75 shadow-[0_0_10px_#22c55e]"
                style={{ width: `${Math.round(gestureProgress * 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Real-time Recognition Status Line */}
        <div className="w-full mt-2.5 flex items-center justify-between px-3 py-2 bg-neutral-900 border border-neutral-800 rounded-xl">
          <div className="flex items-center gap-2 text-xs sm:text-sm font-mono">
            {recognitionResult.handCount === 0 ? (
              <span className="text-neutral-500 italic">Đang chờ tay...</span>
            ) : recognitionResult.isConfident ? (
              <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#22c55e]" />
                {recognitionResult.statusText}
              </span>
            ) : (
              <span className="text-amber-400 font-medium flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                Không chắc chắn ({Math.round(recognitionResult.confidence * 100)}%)
              </span>
            )}
          </div>

          {/* Fast status indicator */}
          <div className="text-[11px] font-mono text-neutral-400">
            {isCoolingDown ? (
              <span className="text-neutral-500">Chuyển tiếp...</span>
            ) : gestureProgress > 0 ? (
              <span className="text-emerald-400 font-bold">
                Giữ ({Math.round(gestureProgress * sensitivityConfig.stabilityFrames)}/{sensitivityConfig.stabilityFrames})
              </span>
            ) : (
              <span className="text-neutral-600">
                Ngưỡng: {Math.round(sensitivityConfig.confidenceThreshold * 100)}%
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Translation Sentence Box */}
      <div className="w-full bg-neutral-900 border border-neutral-800 rounded-2xl p-3.5 sm:p-5 space-y-3 shadow-xl">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="text-xs font-mono uppercase tracking-wider text-neutral-400 font-bold">
            Kết quả ghép câu thời gian thực
          </span>
          <div className="flex items-center gap-1.5">
            {sentenceTokens.length > 0 && (
              <>
                <button
                  onClick={handleSpeak}
                  title="Đọc to câu dịch"
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium rounded-lg transition-colors cursor-pointer"
                >
                  <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Đọc to</span>
                </button>

                <button
                  onClick={handleClearSentence}
                  title="Xoá toàn bộ câu"
                  className="flex items-center gap-1 px-2.5 py-1.5 bg-neutral-800 hover:bg-rose-950/60 hover:text-rose-400 hover:border-rose-800/40 text-neutral-400 text-xs font-medium rounded-lg border border-transparent transition-colors cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Xoá</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Display Raw Sentence: Câu dịch: [chuỗi nhãn đã ghép, cách nhau bằng dấu cách] */}
        <div className="min-h-[56px] flex flex-wrap items-center gap-2 p-3 bg-neutral-950 rounded-xl border border-neutral-800/80">
          <span className="text-xs font-mono uppercase text-emerald-500/80 mr-1 select-none font-semibold">
            Câu dịch:
          </span>

          {sentenceTokens.length === 0 ? (
            <span className="text-xs sm:text-sm text-neutral-600 italic font-sans">
              Chưa có ký hiệu nào được chốt. Hãy giữ cử chỉ tay trước camera (~0.15s) để ghép câu.
            </span>
          ) : (
            sentenceTokens.map((token, idx) => (
              <span
                key={`${token}-${idx}`}
                className="group inline-flex items-center gap-1.5 text-sm sm:text-base font-bold text-white font-mono bg-neutral-800/90 border border-emerald-500/30 px-2.5 py-1 rounded-lg shadow-sm tracking-wide transition-all hover:border-emerald-400"
              >
                <span>{token}</span>
                <button
                  onClick={() => handleRemoveToken(idx)}
                  title="Xoá từ này"
                  className="text-neutral-500 group-hover:text-rose-400 hover:bg-neutral-700 rounded-full p-0.5 transition-colors cursor-pointer"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))
          )}
        </div>

        {/* Optional: Gemini AI Sentence Smoother */}
        <div className="pt-2 border-t border-neutral-800/60 flex flex-col space-y-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-xs font-medium text-neutral-300">
                Làm mượt câu (Gemini AI)
              </span>
              <span className="text-[10px] text-neutral-500 font-mono hidden sm:inline">
                [Tùy chọn · Giữ nguyên ý gốc]
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const nextState = !isSmootherEnabled;
                  setIsSmootherEnabled(nextState);
                  if (nextState && sentenceTokens.length > 0 && !smoothedText) {
                    handleSmoothSentence();
                  }
                }}
                className={`text-xs px-2.5 py-1 rounded-md font-semibold transition-colors cursor-pointer ${
                  isSmootherEnabled
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                    : 'bg-neutral-800 text-neutral-400 hover:text-white'
                }`}
              >
                {isSmootherEnabled ? 'Đang bật' : 'Bật tính năng'}
              </button>

              {isSmootherEnabled && sentenceTokens.length > 0 && (
                <button
                  onClick={handleSmoothSentence}
                  disabled={isSmoothingLoading}
                  className="text-xs px-2.5 py-1 bg-emerald-500 text-neutral-950 font-bold rounded-md hover:bg-emerald-400 transition-colors cursor-pointer disabled:opacity-50"
                >
                  {isSmoothingLoading ? 'Đang xử lý...' : 'Làm mượt'}
                </button>
              )}
            </div>
          </div>

          {/* Smoothed Result Display (parallel to raw tokens) */}
          {isSmootherEnabled && (
            <div className="p-3 bg-neutral-950/60 rounded-xl border border-neutral-800 space-y-1">
              <span className="text-[10px] font-mono text-neutral-400 uppercase">
                Câu tiếng Việt tự nhiên:
              </span>
              {isSmoothingLoading ? (
                <p className="text-xs text-neutral-500 animate-pulse">
                  Đang làm mượt câu với Gemini API...
                </p>
              ) : smoothedText ? (
                <p className="text-sm sm:text-base font-semibold text-emerald-300">
                  {smoothedText}
                </p>
              ) : (
                <p className="text-xs text-neutral-600 italic">
                  Bấm "Làm mượt" sau khi hoàn tất chuỗi ký hiệu để sinh câu tiếng Việt tự nhiên.
                </p>
              )}
              {smoothingError && (
                <p className="text-[10px] text-neutral-500 italic">{smoothingError}</p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Cẩm nang 9 Cử chỉ Ngôn ngữ Ký hiệu Việt Nam */}
      <div className="w-full bg-neutral-900/60 border border-neutral-800 rounded-2xl p-3.5 sm:p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-1.5">
              <span>📖</span>
              <span>Cẩm nang 9 Cử chỉ NNKH chuẩn</span>
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/40 font-mono">
              Đã rà soát & tối ưu 100%
            </span>
          </div>
          <span className="text-[10px] text-neutral-500 font-mono hidden sm:inline">
            Bấm vào cử chỉ để thử nghiệm ghép câu
          </span>
        </div>

        <div className="grid grid-cols-3 sm:grid-cols-3 md:grid-cols-5 gap-2">
          {[
            { label: 'TÔI', icon: '☝️', desc: 'Chỉ ngón trỏ vào ngực' },
            { label: 'BẠN', icon: '👉', desc: 'Chỉ ngón trỏ về phía trước' },
            { label: 'KHỎE', icon: '👍', desc: 'Ngón cái giơ lên dứt khoát' },
            { label: 'CẢM_ƠN', icon: '✋', desc: 'Tay phẳng khép ngón từ cằm ra' },
            { label: 'XIN_CHÀO', icon: '🙋', desc: 'Bàn tay mở vẫy chào' },
            { label: 'TẠM_BIỆT', icon: '🖐️', desc: '5 ngón xòe rộng vẫy chào' },
            { label: 'BÂY_GIỜ', icon: '🫳', desc: 'Tay gập nhấn xuống (1/2 tay)' },
            { label: 'CẢM_THẤY', icon: '🫱', desc: 'Lòng bàn tay vuốt lên ngực' },
            { label: 'VUI_VẺ', icon: '🙌', desc: 'Hai tay mở vỗ nhẹ ở ngực' },
          ].map((item) => (
            <button
              key={item.label}
              onClick={() => handleSimulateGesture(item.label)}
              className="flex flex-col items-start p-2 rounded-xl bg-neutral-950 border border-neutral-800 hover:border-emerald-500/60 transition-all text-left group cursor-pointer active:scale-95 shadow-sm"
            >
              <div className="flex items-center justify-between w-full mb-1">
                <span className="text-base">{item.icon}</span>
                <span className="text-[10px] font-mono text-emerald-400 font-bold group-hover:underline">
                  {item.label}
                </span>
              </div>
              <p className="text-[9px] text-neutral-400 leading-tight line-clamp-2">
                {item.desc}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Quick Test / Gesture Simulator Bar */}
      <div className="w-full bg-neutral-900/50 border border-neutral-800/60 rounded-xl p-3 space-y-2">
        <div className="flex items-center justify-between flex-wrap gap-1">
          <span className="text-[11px] font-mono uppercase text-neutral-400 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-neutral-400" />
            Ký hiệu mẫu nhanh (Nhấn để ghép chuỗi tức thì)
          </span>
          <span className="text-[10px] text-neutral-500">
            Tối ưu cho chuỗi: TÔI BÂY_GIỜ CẢM_THẤY VUI_VẺ
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {distinctLabels.map((lbl) => (
            <button
              key={lbl}
              onClick={() => handleSimulateGesture(lbl)}
              className="px-2.5 py-1 bg-neutral-800 hover:bg-neutral-700 hover:text-emerald-400 text-neutral-300 text-xs font-mono font-medium rounded-lg border border-neutral-700/60 transition-colors cursor-pointer active:scale-95"
            >
              + {lbl}
            </button>
          ))}
        </div>
      </div>

      {/* PERMANENT FIXED WARNING - REQUIRED: CANNOT BE HIDDEN */}
      <div className="w-full bg-neutral-950 border border-amber-900/40 rounded-xl p-3 flex items-start gap-2.5 text-amber-400/90 text-xs select-none">
        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
        <p className="leading-relaxed">
          <strong className="text-amber-300">Cảnh báo:</strong> Kết quả tham khảo — không thay thế phiên dịch viên chuyên nghiệp trong tình huống quan trọng (y tế, pháp lý).
        </p>
      </div>
    </div>
  );
};
