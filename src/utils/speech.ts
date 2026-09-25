/**
 * Web Speech API text-to-speech utility for Vietnamese
 */
export function speakVietnamese(text: string): boolean {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
    console.warn('Speech synthesis not supported on this browser');
    return false;
  }

  try {
    window.speechSynthesis.cancel(); // Cancel any ongoing speech

    // Format sign tokens into readable speech (replace underscores with spaces)
    const readable = text.replace(/_/g, ' ');
    const utterance = new SpeechSynthesisUtterance(readable);
    utterance.lang = 'vi-VN';
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    // Try finding Vietnamese voice if available
    const voices = window.speechSynthesis.getVoices();
    const viVoice = voices.find(
      (v) => v.lang.startsWith('vi') || v.name.toLowerCase().includes('vietnam')
    );
    if (viVoice) {
      utterance.voice = viVoice;
    }

    window.speechSynthesis.speak(utterance);
    return true;
  } catch (err) {
    console.error('Speech synthesis error:', err);
    return false;
  }
}
