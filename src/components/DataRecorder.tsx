import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Database,
  Plus,
  Trash2,
  AlertTriangle,
  CheckCircle,
  Download,
  Upload,
  RotateCcw,
  Camera,
  Layers,
  Info,
} from 'lucide-react';
import { SignSample, Landmark } from '../types';
import {
  getHandLandmarker,
  drawLandmarksOnCanvas,
} from '../utils/mediaPipeService';
import {
  normalizeHandLandmarks,
} from '../utils/handNormalization';
import { resetToDefaultDataset } from '../data/defaultSignDataset';

interface DataRecorderProps {
  dataset: SignSample[];
  onUpdateDataset: (newDataset: SignSample[]) => void;
  availableProfiles: string[];
  activeProfile: string;
  onProfileChange: (p: string) => void;
}

export const DataRecorder: React.FC<DataRecorderProps> = ({
  dataset,
  onUpdateDataset,
  availableProfiles,
  activeProfile,
  onProfileChange,
}) => {
  // Profile state: Must choose or create a profile!
  const [selectedProfile, setSelectedProfile] = useState<string>(
    activeProfile === 'global' ? availableProfiles[0] || 'khang' : activeProfile
  );
  const [newProfileInput, setNewProfileInput] = useState<string>('');

  // Sign label input: Convention UPPERCASE with UNDERSCORE
  const [labelInput, setLabelInput] = useState<string>('XIN_CHÀO');

  // Recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordedFramesCount, setRecordedFramesCount] = useState<number>(0);
  const recordedLandmarkFramesRef = useRef<{
    handCount: number;
    norm1: number[];
    norm2?: number[];
    interWristOffset?: { dx: number; dy: number; dz: number };
  }[]>([]);

  // Video / Canvas ref for recorder camera preview
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const isRecordingRef = useRef<boolean>(false);
  const animFrameIdRef = useRef<number | null>(null);

  // Status message
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  const showFeedback = (msg: string) => {
    setFeedbackMessage(msg);
    setTimeout(() => {
      setFeedbackMessage((prev) => (prev === msg ? null : prev));
    }, 3000);
  };

  // Keep isRecordingRef in sync for animation loop
  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Start Camera
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError('Trình duyệt không hỗ trợ truy cập camera.');
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
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
      console.error('Camera error in recorder:', err);
      setCameraError('Không thể mở camera. Vui lòng cấp quyền truy cập.');
      setCameraActive(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }, []);

  // Initialize camera for recorder
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [startCamera, stopCamera]);

  // Video processing & live capture loop
  useEffect(() => {
    let isRunning = true;

    async function processLoop() {
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

            if (ctx) {
              drawLandmarksOnCanvas(ctx, handsLandmarks, canvas.width, canvas.height, true);
            }

            // If recording is active and hands are in view, accumulate frames!
            if (isRecordingRef.current && handsLandmarks.length > 0) {
              const hCount = handsLandmarks.length >= 2 ? 2 : 1;
              const norm1 = normalizeHandLandmarks(handsLandmarks[0]);
              const norm2 = hCount === 2 ? normalizeHandLandmarks(handsLandmarks[1]) : null;

              if (norm1) {
                let interWristOffset = undefined;
                if (norm2) {
                  const avgScale = (norm1.scale + norm2.scale) / 2 || 1;
                  interWristOffset = {
                    dx: (norm2.wrist.x - norm1.wrist.x) / avgScale,
                    dy: (norm2.wrist.y - norm1.wrist.y) / avgScale,
                    dz: ((norm2.wrist.z ?? 0) - (norm1.wrist.z ?? 0)) / avgScale,
                  };
                }

                recordedLandmarkFramesRef.current.push({
                  handCount: hCount,
                  norm1: norm1.points,
                  norm2: norm2?.points,
                  interWristOffset,
                });

                setRecordedFramesCount(recordedLandmarkFramesRef.current.length);
              }
            }
          } catch (e) {
            console.error('Recorder detection error:', e);
          }
        }
      }

      if (isRunning) {
        animFrameIdRef.current = requestAnimationFrame(processLoop);
      }
    }

    animFrameIdRef.current = requestAnimationFrame(processLoop);

    return () => {
      isRunning = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, []);

  // Format label string (UPPERCASE, spaces to underscore)
  const handleLabelChange = (val: string) => {
    const formatted = val
      .toUpperCase()
      .replace(/\s+/g, '_')
      .replace(/[^A-Z0-9_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴÝỶỸ]/g, '');
    setLabelInput(formatted);
  };

  // Add new Profile
  const handleAddProfile = () => {
    const clean = newProfileInput.trim().toLowerCase();
    if (!clean) return;
    if (!availableProfiles.includes(clean)) {
      onProfileChange(clean);
    }
    setSelectedProfile(clean);
    setNewProfileInput('');
    showFeedback(`Đã tạo Profile: ${clean}`);
  };

  // Start / Stop Recording
  const startRecording = () => {
    if (!selectedProfile) {
      alert('Vui lòng chọn hoặc tạo Profile trước khi ghi!');
      return;
    }
    if (!labelInput.trim()) {
      alert('Vui lòng nhập nhãn ký hiệu!');
      return;
    }
    recordedLandmarkFramesRef.current = [];
    setRecordedFramesCount(0);
    setIsRecording(true);
  };

  const finishRecording = () => {
    if (!isRecording) return;
    setIsRecording(false);

    const frames = recordedLandmarkFramesRef.current;
    if (frames.length < 3) {
      showFeedback('Mẫu quá ngắn hoặc không phát hiện bàn tay. Hãy giữ phím và đưa tay trước camera.');
      return;
    }

    // Determine majority handCount
    const hand2Count = frames.filter((f) => f.handCount === 2).length;
    const finalHandCount: 1 | 2 = hand2Count > frames.length / 2 ? 2 : 1;

    // Filter frames matching dominant hand count
    const validFrames = frames.filter((f) => f.handCount === finalHandCount);
    if (validFrames.length === 0) return;

    // Compute average normalized landmark vector across recorded frames to eliminate jitter
    const avgHand1 = new Array(63).fill(0);
    for (const f of validFrames) {
      for (let i = 0; i < 63; i++) {
        avgHand1[i] += f.norm1[i] / validFrames.length;
      }
    }

    let avgHand2: number[] | undefined = undefined;
    let avgInterWrist = undefined;

    if (finalHandCount === 2) {
      avgHand2 = new Array(63).fill(0);
      let dxSum = 0, dySum = 0, dzSum = 0;
      let count2 = 0;

      for (const f of validFrames) {
        if (f.norm2) {
          count2++;
          for (let i = 0; i < 63; i++) {
            avgHand2[i] += f.norm2[i];
          }
          if (f.interWristOffset) {
            dxSum += f.interWristOffset.dx;
            dySum += f.interWristOffset.dy;
            dzSum += f.interWristOffset.dz;
          }
        }
      }

      if (count2 > 0) {
        for (let i = 0; i < 63; i++) {
          avgHand2[i] /= count2;
        }
        avgInterWrist = {
          dx: dxSum / count2,
          dy: dySum / count2,
          dz: dzSum / count2,
        };
      }
    }

    const newSample: SignSample = {
      id: `sample_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      label: labelInput,
      profile: selectedProfile,
      handCount: finalHandCount,
      hand1: avgHand1,
      hand2: avgHand2,
      interWristOffset: avgInterWrist,
      createdAt: Date.now(),
    };

    const nextDataset = [...dataset, newSample];
    onUpdateDataset(nextDataset);
    showFeedback(`Đã lưu mẫu "${labelInput}" (${validFrames.length} frames) cho profile "${selectedProfile}"!`);
  };

  // Keyboard shortcut: Hold Spacebar to record
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && e.target === document.body) {
        e.preventDefault();
        startRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        finishRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [selectedProfile, labelInput, dataset, isRecording]);

  // Delete single sample
  const handleDeleteSample = (sampleId: string) => {
    const next = dataset.filter((s) => s.id !== sampleId);
    onUpdateDataset(next);
    showFeedback('Đã xoá 1 mẫu.');
  };

  // Delete entire label
  const handleDeleteLabel = (label: string) => {
    if (confirm(`Bạn có chắc muốn xoá toàn bộ mẫu của ký hiệu "${label}" không?`)) {
      const next = dataset.filter((s) => s.label !== label);
      onUpdateDataset(next);
      showFeedback(`Đã xoá ký hiệu ${label}.`);
    }
  };

  // Reset to default dataset
  const handleResetDataset = () => {
    if (confirm('Đặt lại toàn bộ tập dữ liệu về mặc định (9 ký hiệu mẫu)?')) {
      const def = resetToDefaultDataset();
      onUpdateDataset(def);
      showFeedback('Đã đặt lại tập dữ liệu mẫu gốc.');
    }
  };

  // Export dataset to JSON
  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(dataset, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `voice2hand_dataset_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Import dataset from JSON
  const handleImportJSON = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (Array.isArray(parsed) && parsed.length > 0) {
          onUpdateDataset(parsed);
          showFeedback(`Đã nhập thành công ${parsed.length} mẫu.`);
        } else {
          alert('Tập tin JSON không đúng định dạng tập mẫu.');
        }
      } catch (err) {
        alert('Lỗi đọc tập tin JSON.');
      }
    };
    reader.readAsText(file);
  };

  // Group dataset by label for the table view
  const groupedByLabel: Record<string, { total: number; byProfile: Record<string, number>; sampleIds: string[] }> = {};
  for (const s of dataset) {
    if (!groupedByLabel[s.label]) {
      groupedByLabel[s.label] = { total: 0, byProfile: {}, sampleIds: [] };
    }
    groupedByLabel[s.label].total++;
    groupedByLabel[s.label].byProfile[s.profile] =
      (groupedByLabel[s.label].byProfile[s.profile] || 0) + 1;
    groupedByLabel[s.label].sampleIds.push(s.id);
  }

  const labelList = Object.keys(groupedByLabel).sort();

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-4 space-y-6">
      {/* Toast Feedback */}
      {feedbackMessage && (
        <div className="fixed top-16 right-4 z-50 bg-neutral-900 border border-emerald-500/40 text-emerald-400 px-4 py-2 rounded-lg shadow-xl text-xs font-mono flex items-center gap-2 animate-bounce">
          <CheckCircle className="w-4 h-4 text-emerald-400" />
          <span>{feedbackMessage}</span>
        </div>
      )}

      {/* Admin Title & Info */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-neutral-800 pb-4">
        <div>
          <h2 className="text-xl font-black tracking-wide text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-400" />
            DATA RECORDER — THU THẬP & QUẢN LÝ MẪU KÝ HIỆU
          </h2>
          <p className="text-xs text-neutral-400 mt-1">
            Ghi và chuẩn hoá landmark bàn tay để huấn luyện mô hình k-NN cục bộ.
          </p>
        </div>

        {/* Global Dataset Action buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 text-xs font-medium rounded-lg border border-neutral-800 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-neutral-400" />
            <span>Xuất JSON</span>
          </button>

          <label className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 text-xs font-medium rounded-lg border border-neutral-800 cursor-pointer transition-colors">
            <Upload className="w-3.5 h-3.5 text-neutral-400" />
            <span>Nhập JSON</span>
            <input
              type="file"
              accept=".json"
              onChange={handleImportJSON}
              className="hidden"
            />
          </label>

          <button
            onClick={handleResetDataset}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral-900 hover:bg-rose-950/60 hover:text-rose-400 text-neutral-400 text-xs font-medium rounded-lg border border-neutral-800 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Đặt lại gốc</span>
          </button>
        </div>
      </div>

      {/* Top Section: Profile & Label Input + Camera Recorder */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Input Form */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 sm:p-5 space-y-4">
          {/* 1. Mandatory Profile Selection */}
          <div className="space-y-1.5">
            <label className="block text-xs font-mono uppercase tracking-wider text-emerald-400 font-bold flex items-center justify-between">
              <span>1. Người ký (Profile bắt buộc)</span>
              <span className="text-[10px] text-neutral-500 font-normal">
                Cần đa dạng người ký
              </span>
            </label>
            <div className="flex items-center gap-2">
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                className="w-full bg-neutral-950 border border-neutral-700 text-neutral-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500 font-medium"
              >
                {availableProfiles.map((p) => (
                  <option key={p} value={p}>
                    Profile: {p}
                  </option>
                ))}
              </select>
            </div>

            {/* Create new Profile row */}
            <div className="flex items-center gap-2 pt-1">
              <input
                type="text"
                placeholder="Tạo profile mới (vd: hieu, mai)..."
                value={newProfileInput}
                onChange={(e) => setNewProfileInput(e.target.value)}
                className="w-full bg-neutral-950 border border-neutral-800 text-neutral-300 text-xs rounded-lg px-3 py-1.5 focus:outline-none focus:border-emerald-500"
              />
              <button
                type="button"
                onClick={handleAddProfile}
                className="px-3 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-semibold rounded-lg shrink-0 transition-colors"
              >
                + Thêm
              </button>
            </div>
          </div>

          {/* 2. Sign Label Input */}
          <div className="space-y-1.5">
            <label className="block text-xs font-mono uppercase tracking-wider text-emerald-400 font-bold flex items-center justify-between">
              <span>2. Nhãn ký hiệu (Convention: CHỮ_HOA)</span>
              <span className="text-[10px] text-neutral-500 font-normal">
                Tự động nối bằng gạch dưới
              </span>
            </label>
            <input
              type="text"
              value={labelInput}
              onChange={(e) => handleLabelChange(e.target.value)}
              placeholder="VD: TÔI, BÂY_GIỜ, CẢM_THẤY, VUI_VẺ..."
              className="w-full bg-neutral-950 border border-neutral-700 text-white font-mono font-bold text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500 tracking-wider"
            />
          </div>

          {/* 3. Record Button & Shortcut */}
          <div className="pt-2 space-y-2">
            <button
              onMouseDown={startRecording}
              onMouseUp={finishRecording}
              onTouchStart={startRecording}
              onTouchEnd={finishRecording}
              className={`w-full py-3.5 px-4 rounded-xl font-bold font-mono tracking-wider text-sm transition-all flex items-center justify-center gap-2 select-none shadow-lg cursor-pointer ${
                isRecording
                  ? 'bg-rose-600 text-white animate-pulse shadow-rose-900/50 scale-[0.99]'
                  : 'bg-emerald-500 hover:bg-emerald-400 text-neutral-950 shadow-emerald-950/40'
              }`}
            >
              <div
                className={`w-3 h-3 rounded-full ${
                  isRecording ? 'bg-white' : 'bg-neutral-950'
                }`}
              />
              <span>
                {isRecording
                  ? `RECORDING: ${recordedFramesCount} frames`
                  : 'GIỮ CHUỘT / PHÍM SPACE ĐỂ GHI MẪU'}
              </span>
            </button>

            <p className="text-[11px] text-neutral-500 text-center font-mono">
              💡 Mẹo: Nhấn và GIỮ phím <strong>Space</strong> khi đang làm ký hiệu, sau đó nhả phím để lưu mẫu.
            </p>
          </div>
        </div>

        {/* Right: Camera Feed with Landmark Skeleton */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 flex flex-col items-center justify-between space-y-3">
          <div className="relative w-full aspect-[4/3] bg-neutral-950 rounded-xl overflow-hidden border border-neutral-800">
            <video
              ref={videoRef}
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover -scale-x-100"
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10"
            />

            {/* Live recording banner */}
            {isRecording && (
              <div className="absolute top-2 left-2 z-20 flex items-center gap-1.5 bg-rose-600/90 text-white px-2.5 py-1 rounded-md text-xs font-mono font-bold animate-pulse">
                <span className="w-2 h-2 rounded-full bg-white" />
                RECORDING: {recordedFramesCount} frames
              </div>
            )}

            {!cameraActive && (
              <div className="absolute inset-0 z-30 bg-neutral-950 flex flex-col items-center justify-center p-4 text-center">
                <Camera className="w-8 h-8 text-neutral-600 mb-2" />
                <p className="text-xs text-neutral-400">
                  {cameraError || 'Đang mở camera...'}
                </p>
              </div>
            )}
          </div>

          <div className="w-full flex items-center justify-between text-[11px] font-mono text-neutral-400 px-1">
            <span>Đang gán cho: <strong>{selectedProfile}</strong></span>
            <span>Ký hiệu: <strong className="text-emerald-400">{labelInput || '---'}</strong></span>
          </div>
        </div>
      </div>

      {/* Dataset Summary Table */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>Danh sách ký hiệu hiện có ({labelList.length} nhãn)</span>
            <span className="text-xs text-neutral-500 font-normal">
              · Tổng {dataset.length} mẫu
            </span>
          </h3>
          <span className="text-[11px] text-neutral-400 font-mono">
            Tiêu chuẩn: ≥ 5 mẫu & ≥ 2 profiles
          </span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 font-mono text-[11px] uppercase">
                <th className="pb-2.5 px-3">Ký hiệu (Label)</th>
                <th className="pb-2.5 px-3">Phân bổ Profile</th>
                <th className="pb-2.5 px-3">Tổng số mẫu</th>
                <th className="pb-2.5 px-3">Đánh giá chất lượng</th>
                <th className="pb-2.5 px-3 text-right">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/60 font-medium">
              {labelList.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-neutral-500 italic">
                    Chưa có mẫu nào trong hệ thống. Hãy thực hiện ghi mẫu ở trên.
                  </td>
                </tr>
              ) : (
                labelList.map((lbl) => {
                  const stat = groupedByLabel[lbl];
                  const profileKeys = Object.keys(stat.byProfile);
                  const isLowSamples = stat.total < 5;
                  const isSingleProfile = profileKeys.length === 1;

                  return (
                    <tr key={lbl} className="hover:bg-neutral-800/30 transition-colors">
                      {/* Label */}
                      <td className="py-3 px-3">
                        <span className="font-mono font-bold text-emerald-400 bg-neutral-950 px-2 py-1 rounded border border-neutral-800">
                          {lbl}
                        </span>
                      </td>

                      {/* Profiles breakdown */}
                      <td className="py-3 px-3">
                        <div className="flex flex-wrap gap-1">
                          {profileKeys.map((p) => (
                            <span
                              key={p}
                              className="px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 text-[10px] font-mono"
                            >
                              {p}: <strong>{stat.byProfile[p]}</strong>
                            </span>
                          ))}
                        </div>
                      </td>

                      {/* Total */}
                      <td className="py-3 px-3 font-mono font-bold text-white">
                        {stat.total}
                      </td>

                      {/* Quality Badge */}
                      <td className="py-3 px-3">
                        <div className="flex flex-col gap-1 items-start">
                          {isLowSamples && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-950/60 text-amber-400 border border-amber-800/50">
                              <AlertTriangle className="w-3 h-3" />
                              Cần thêm mẫu (&lt;5)
                            </span>
                          )}

                          {isSingleProfile && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-950/60 text-rose-400 border border-rose-800/50">
                              <AlertTriangle className="w-3 h-3" />
                              Chỉ 1 profile (Rủi ro lệch người ký)
                            </span>
                          )}

                          {!isLowSamples && !isSingleProfile && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/50">
                              <CheckCircle className="w-3 h-3" />
                              Đạt chuẩn đa người ký
                            </span>
                          )}
                        </div>
                      </td>

                      {/* Actions */}
                      <td className="py-3 px-3 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleDeleteSample(stat.sampleIds[stat.sampleIds.length - 1])}
                            title="Xoá mẫu vừa ghi gần nhất"
                            className="text-[11px] text-neutral-400 hover:text-amber-400 px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 transition-colors"
                          >
                            Xoá 1 mẫu
                          </button>
                          <button
                            onClick={() => handleDeleteLabel(lbl)}
                            title="Xoá toàn bộ mẫu của ký hiệu này"
                            className="p-1 rounded text-neutral-500 hover:text-rose-400 hover:bg-rose-950/50 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
