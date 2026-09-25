import React from 'react';
import { Video, Database } from 'lucide-react';

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
}) => {
  return (
    <header className="border-b border-neutral-800 bg-neutral-950/90 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-5xl mx-auto px-3 sm:px-4 py-2.5 flex flex-col sm:flex-row items-center justify-between gap-2.5">
        {/* Logo & Title */}
        <div className="flex items-center justify-between w-full sm:w-auto">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
            <h1 className="text-lg sm:text-xl font-black tracking-wider text-white">
              VOICE2HAND
            </h1>
            <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-neutral-800 text-emerald-400 border border-emerald-500/20">
              NNKH
            </span>
          </div>
          <span className="text-[11px] text-neutral-400 font-medium sm:hidden">
            Dịch Ký Hiệu VN
          </span>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 p-1 bg-neutral-900 border border-neutral-800 rounded-lg w-full sm:w-auto justify-center">
          <button
            onClick={() => onTabChange('translate')}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              currentTab === 'translate'
                ? 'bg-emerald-500 text-neutral-950 shadow-sm font-bold'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>Màn hình dịch</span>
          </button>

          <button
            onClick={() => onTabChange('recorder')}
            className={`flex-1 sm:flex-initial flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              currentTab === 'recorder'
                ? 'bg-emerald-500 text-neutral-950 shadow-sm font-bold'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Ghi dữ liệu</span>
            <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded-full bg-neutral-800 text-neutral-300 font-mono">
              {datasetCount}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
};
