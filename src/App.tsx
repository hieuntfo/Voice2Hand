import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { LiveTranslate } from './components/LiveTranslate';
import { DataRecorder } from './components/DataRecorder';
import { SignSample } from './types';
import {
  loadStoredDataset,
  saveStoredDataset,
  STORAGE_KEY_PROFILE,
} from './data/defaultSignDataset';

export default function App() {
  // Load dataset from localStorage
  const [dataset, setDataset] = useState<SignSample[]>(() => loadStoredDataset());

  // Active profile: 'global' or profile name e.g. 'khang'
  const [activeProfile, setActiveProfile] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEY_PROFILE) || 'global';
    } catch {
      return 'global';
    }
  });

  // Current Screen / Route: 'translate' or 'recorder'
  const [currentTab, setCurrentTab] = useState<'translate' | 'recorder'>(() => {
    if (typeof window !== 'undefined') {
      const path = window.location.pathname;
      if (path.includes('/admin/record') || window.location.hash.includes('record')) {
        return 'recorder';
      }
    }
    return 'translate';
  });

  // Keep active profile synced with localStorage
  const handleProfileChange = (profile: string) => {
    setActiveProfile(profile);
    try {
      localStorage.setItem(STORAGE_KEY_PROFILE, profile);
    } catch (e) {
      console.error(e);
    }
  };

  // Update dataset (add, remove, reset)
  const handleUpdateDataset = (newDataset: SignSample[]) => {
    setDataset(newDataset);
    saveStoredDataset(newDataset);
  };

  // Unique list of profiles across the dataset
  const availableProfiles = Array.from(
    new Set(dataset.map((s) => s.profile.toLowerCase()))
  ).sort();

  return (
    <div className="min-h-screen bg-[#090d16] text-neutral-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-neutral-950">
      {/* Top Navigation & App Branding */}
      <Header
        currentTab={currentTab}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          if (typeof window !== 'undefined') {
            window.location.hash = tab === 'recorder' ? '#record' : '#translate';
          }
        }}
        datasetCount={dataset.length}
        profileCount={availableProfiles.length}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col items-center justify-start py-4">
        {currentTab === 'translate' ? (
          <LiveTranslate
            dataset={dataset}
            activeProfile={activeProfile}
            onProfileChange={handleProfileChange}
            availableProfiles={availableProfiles}
          />
        ) : (
          <DataRecorder
            dataset={dataset}
            onUpdateDataset={handleUpdateDataset}
            availableProfiles={availableProfiles}
            activeProfile={activeProfile}
            onProfileChange={handleProfileChange}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-neutral-900 bg-neutral-950/60 py-4 text-center text-xs text-neutral-500">
        <div className="max-w-4xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Voice2Hand · Hệ thống hỗ trợ giao tiếp Ngôn ngữ Ký hiệu Việt Nam (NNKH)</span>
          <span className="font-mono text-[11px] text-neutral-600">
            Client-side Landmark Inference · MediaPipe & k-NN
          </span>
        </div>
      </footer>
    </div>
  );
}
