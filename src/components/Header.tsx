import React from 'react';
import { Video, Database, Sparkles, Volume2 } from 'lucide-react';

interface HeaderProps {
  currentTab: 'translate' | 'recorder';
  onTabChange: (tab: 'translate' | 'recorder') => void;
  datasetCount: number;
  profileCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  datasetCount,
  profileCount,
}) => {
  return (
    <header className="border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-6xl mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Logo & Title */}
        <div className="text-center sm:text-left">
          <div className="flex items-center justify-center sm:justify-start gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
            <h1 className="text-xl sm:text-2xl font-black tracking-wider text-white">
              VOICE2HAND
            </h1>
            <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-neutral-800 text-emerald-400 border border-emerald-500/20">
              Client AI
            </span>
          </div>
          <p className="text-xs text-neutral-400 font-medium tracking-wide">
            Sign Language Communication · Ngôn ngữ Ký hiệu Việt Nam (NNKH)
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1.5 p-1 bg-neutral-900 border border-neutral-800 rounded-lg">
          <button
            onClick={() => onTabChange('translate')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              currentTab === 'translate'
                ? 'bg-emerald-500 text-neutral-950 shadow-sm'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>Màn hình dịch (Live)</span>
          </button>

          <button
            onClick={() => onTabChange('recorder')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
              currentTab === 'recorder'
                ? 'bg-emerald-500 text-neutral-950 shadow-sm'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Ghi dữ liệu (Data Recorder)</span>
            <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded-full bg-neutral-800 text-neutral-300">
              {datasetCount}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
