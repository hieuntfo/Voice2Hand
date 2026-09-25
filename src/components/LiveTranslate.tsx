import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Volume2,
  Trash2,
  Sparkles,
  Camera,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Users,
  Eye,
  Info,
  Layers,
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
  CONFIDENCE_THRESHOLD,
  FrameInputData,
} from '../utils/knnClassifier';
import { SentenceAggregator } from '../utils/sentenceBuilder';
import { speakVietnamese } from '../utils/speech';

interface LiveTranslateProps {
  dataset: SignSample[];
  activeProfile: string;
  onProfileChange: (profile: string) => void;
  availableProfiles: string[];
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

  // Camera & Model state
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState<boolean>(true);

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
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [gestureProgress, setGestureProgress] = useState<number>(0);
  const [isCoolingDown, setIsCoolingDown] = useState<boolean>(false);

  // Optional: Gemini Smoothed Sentence
  const [isSmootherEnabled, setIsSmootherEnabled] = useState<boolean>(false);
  const [smoothedText, setSmoothedText] = useState<string | null>(null);
  const [isSmoothingLoading, setIsSmoothingLoading] = useState<boolean>(false);
  const [smoothingError, setSmoothingError] = useState<string | null>(null);

  // Temporal aggregator ref
  const aggregatorRef = useRef<SentenceAggregator>(new SentenceAggregator());

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
    }, 2400);
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

  // Main Detection Loop
  useEffect(() => {
    let isRunning = true;

    async function processVideoFrame() {
      if (!isRunning) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (
        video &&
        canvas &&
        video.readyState >= 2 &&
        !video.paused &&
        !video.ended
      ) {
        const landmarker = await getHandLandmarker();

        if (landmarker) {
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
                  interWristOffset: {
                    dx: (norm2.wrist.x - norm1.wrist.x) / avgScale,
                    dy: (norm2.wrist.y - norm1.wrist.y) / avgScale,
                    dz: ((norm2.wrist.z ?? 0) - (norm1.wrist.z ?? 0)) / avgScale,
                  },
                };
              }
            }

            // Classify with k-NN
            const recognition = classifyHandGesture(frameInput, dataset, activeProfile);
            setRecognitionResult(recognition);

            // Temporal Segmentation & Sentence building
            const { committedToken, progress, isCoolingDown: cooldown } =
              aggregatorRef.current.processFrame(recognition);

            setGestureProgress(progress);
            setIsCoolingDown(cooldown);

            if (committedToken) {
              setSentenceTokens((prev) => {
                const next = [...prev, committedToken];
                return next;
              });
              showToast(`Đã nhận diện: ${committedToken}`);
            }
          } catch (detErr) {
            console.error('Detection frame error:', detErr);
          }
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
  }, [dataset, activeProfile]);

  // Clear Sentence
  const handleClearSentence = () => {
    setSentenceTokens([]);
    setSmoothedText(null);
    setSmoothingError(null);
    aggregatorRef.current.reset();
    showToast('Đã xoá câu');
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
      // Client fallback: basic capitalization and spaces
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

  // Quick gesture simulator (allows testing the temporal segmentation without webcam)
  const handleSimulateGesture = (label: string) => {
    const simulatedRecognition: RecognitionResult = {
      label,
      confidence: 0.94,
      handCount: ['BÂY_GIỜ', 'VUI_VẺ'].includes(label) ? 2 : 1,
      isConfident: true,
      statusText: `✓ ${label} · confidence 94%`,
      topMatches: [{ label, score: 0.94 }],
    };

    setRecognitionResult(simulatedRecognition);

    // Simulate 8 consecutive frames to satisfy temporal segmentation rule
    for (let i = 0; i < 8; i++) {
      const { committedToken } = aggregatorRef.current.processFrame(simulatedRecognition);
      if (committedToken) {
        setSentenceTokens((prev) => [...prev, committedToken]);
        showToast(`Đã nhận diện: ${committedToken}`);
        break;
      }
    }
  };

  const rawSentenceString = sentenceTokens.join(' ');

  return (
    <div className="flex flex-col items-center w-full max-w-4xl mx-auto px-4 py-4 space-y-4">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed top-16 right-4 z-50 bg-neutral-900 border border-emerald-500/40 text-emerald-400 px-4 py-2 rounded-lg shadow-xl text-xs font-mono flex items-center gap-2 animate-bounce">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Profile & Model Status Bar */}
      <div className="w-full flex flex-col sm:flex-row items-center justify-between bg-neutral-900/90 border border-neutral-800 rounded-xl px-4 py-2.5 gap-2.5">
        {/* Profile Dropdown */}
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Users className="w-4 h-4 text-neutral-400 shrink-0" />
          <span className="text-xs text-neutral-400 font-medium">Chế độ mẫu:</span>
          <select
            value={activeProfile}
            onChange={(e) => onProfileChange(e.target.value)}
            className="bg-neutral-950 border border-neutral-700 text-neutral-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-emerald-500 font-medium cursor-pointer"
          >
            <option value="global">🌐 Global Model (Toàn bộ người ký)</option>
            {availableProfiles.map((p) => (
              <option key={p} value={p}>
                👤 Profile: {p}
              </option>
            ))}
          </select>
        </div>

        {/* Model Status Indicator */}
        <div className="flex items-center gap-2 text-xs font-medium">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-neutral-300">
            {activeProfile === 'global' ? (
              <span className="text-emerald-400">
                ✓ Global Model sẵn sàng · <strong className="text-white">{distinctLabels.length}</strong> ký hiệu
              </span>
            ) : (
              <span className="text-emerald-400">
                ✓ Profile: <strong className="text-white">{activeProfile}</strong> sẵn sàng ·{' '}
                <strong className="text-white">{distinctLabels.length}</strong> ký hiệu
              </span>
            )}
          </span>
        </div>
      </div>

      {/* Centered Camera Container */}
      <div className="w-full max-w-2xl flex flex-col items-center">
        <div className="relative w-full aspect-[4/3] bg-neutral-950 rounded-2xl border-2 border-neutral-800 hover:border-emerald-500/40 transition-colors overflow-hidden shadow-2xl flex items-center justify-center">
          {/* Label "TRANSLATION" at top-left */}
          <div className="absolute top-3 left-3 z-20 flex items-center gap-1.5 bg-neutral-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-neutral-800 text-[11px] font-mono tracking-wider text-emerald-400 font-bold uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            TRANSLATION
          </div>

          {/* Hand Count Indicator at top-right */}
          <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-neutral-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-neutral-800 text-[11px] font-mono text-neutral-400">
            <Layers className="w-3 h-3 text-neutral-400" />
            <span>
              {recognitionResult.handCount === 0
                ? '0 tay'
                : `${recognitionResult.handCount} tay`}
            </span>
          </div>

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
                    className="px-4 py-2 bg-emerald-500 text-neutral-950 font-bold rounded-lg text-xs hover:bg-emerald-400 transition-colors shadow"
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
                    className="px-4 py-2 bg-emerald-500 text-neutral-950 font-bold rounded-lg text-xs hover:bg-emerald-400 transition-colors"
                  >
                    Bật Camera
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Temporal Confirmation Progress Bar */}
          {gestureProgress > 0 && gestureProgress < 1 && (
            <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-neutral-900/80 z-20 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-75"
                style={{ width: `${Math.round(gestureProgress * 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Real-time Recognition Status Line */}
        <div className="w-full mt-3 flex items-center justify-between px-3 py-2 bg-neutral-900 border border-neutral-800 rounded-xl">
          <div className="flex items-center gap-2 text-xs sm:text-sm font-mono">
            {recognitionResult.handCount === 0 ? (
              <span className="text-neutral-500 italic">Đang chờ tay...</span>
            ) : recognitionResult.isConfident ? (
              <span className="text-emerald-400 font-bold flex items-center gap-1.5">
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#22c55e]" />
                {recognitionResult.statusText}
              </span>
            ) : (
              <span className="text-amber-400 font-medium flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-amber-400" />
                Không chắc chắn ({Math.round(recognitionResult.confidence * 100)}%)
              </span>
            )}
          </div>

          {/* Cooldown or Stabilization indicator */}
          <div className="text-[11px] font-mono text-neutral-400">
            {isCoolingDown ? (
              <span className="text-neutral-500">Chờ chuyển động...</span>
            ) : gestureProgress > 0 ? (
              <span className="text-emerald-400">Đang giữ ({Math.round(gestureProgress * 8)}/8)</span>
            ) : (
              <span className="text-neutral-600">Ngưỡng: 60%</span>
            )}
          </div>
        </div>
      </div>

      {/* Main Translation Sentence Box */}
      <div className="w-full bg-neutral-900 border border-neutral-800 rounded-2xl p-4 sm:p-5 space-y-3.5 shadow-xl">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono uppercase tracking-wider text-neutral-400">
            Kết quả ghép câu thời gian thực
          </span>
          <div className="flex items-center gap-2">
            {sentenceTokens.length > 0 && (
              <>
                <button
                  onClick={handleSpeak}
                  title="Đọc to câu dịch"
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium rounded-lg transition-colors cursor-pointer"
                >
                  <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Đọc to</span>
                </button>

                <button
                  onClick={handleClearSentence}
                  title="Xoá câu hiện tại"
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral-800 hover:bg-rose-950/60 hover:text-rose-400 hover:border-rose-800/40 text-neutral-400 text-xs font-medium rounded-lg border border-transparent transition-colors cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Xoá câu</span>
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
            <span className="text-sm text-neutral-600 italic font-sans">
              Chưa có ký hiệu nào được chốt. Hãy thực hiện ký hiệu trước camera (ví dụ: TÔI, BÂY_GIỜ, CẢM_THẤY, VUI_VẺ).
            </span>
          ) : (
            sentenceTokens.map((token, idx) => (
              <span
                key={`${token}-${idx}`}
                className="text-base sm:text-lg font-bold text-white font-mono bg-neutral-800/80 border border-emerald-500/30 px-3 py-1 rounded-lg shadow-sm tracking-wide"
              >
                {token}
              </span>
            ))
          )}
        </div>

        {/* Optional: Gemini AI Sentence Smoother */}
        <div className="pt-2 border-t border-neutral-800/60 flex flex-col space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-medium text-neutral-300">
                Làm mượt câu tự nhiên (Gemini AI)
              </span>
              <span className="text-[10px] text-neutral-500 font-mono">
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
              <span className="text-[11px] font-mono text-neutral-400 uppercase">
                Câu tiếng Việt tự nhiên:
              </span>
              {isSmoothingLoading ? (
                <p className="text-xs text-neutral-500 animate-pulse">
                  Đang làm mượt câu với Gemini API...
                </p>
              ) : smoothedText ? (
                <p className="text-base font-semibold text-emerald-300">
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

      {/* Quick Test / Gesture Simulator Bar */}
      <div className="w-full bg-neutral-900/50 border border-neutral-800/60 rounded-xl p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-mono uppercase text-neutral-400 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-neutral-400" />
            Kiểm tra nhanh ký hiệu mẫu (Mô phỏng chuỗi câu)
          </span>
          <span className="text-[10px] text-neutral-500">
            Bấm để kiểm tra ghép câu "TÔI BÂY_GIỜ CẢM_THẤY VUI_VẺ"
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {distinctLabels.map((lbl) => (
            <button
              key={lbl}
              onClick={() => handleSimulateGesture(lbl)}
              className="px-2.5 py-1 bg-neutral-800 hover:bg-neutral-700 hover:text-emerald-400 text-neutral-300 text-xs font-mono font-medium rounded-lg border border-neutral-700/60 transition-colors cursor-pointer"
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
