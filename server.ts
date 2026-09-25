import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json());

// Initialize Gemini client server-side
const apiKey = process.env.GEMINI_API_KEY || '';
const ai = apiKey
  ? new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    })
  : null;

// API: Smooth sign language sequence into natural Vietnamese sentence
app.post('/api/smooth-sentence', async (req, res) => {
  try {
    const { tokens, rawText } = req.body;
    const inputChain = rawText || (Array.isArray(tokens) ? tokens.join(' ') : '');

    if (!inputChain || typeof inputChain !== 'string' || inputChain.trim().length === 0) {
      return res.status(400).json({ error: 'Chuỗi ký hiệu rỗng' });
    }

    if (!ai) {
      // Local fallback smoothing if API key not configured yet
      const fallback = inputChain
        .toLowerCase()
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
      const capitalized = fallback.charAt(0).toUpperCase() + fallback.slice(1) + '.';
      return res.json({
        smoothedText: capitalized,
        source: 'local_rule',
        note: 'Chuyển đổi theo quy tắc cục bộ (API key chưa cấu hình)',
      });
    }

    const prompt = `Bạn là trợ lý dịch thuật chuyên biệt cho Ngôn ngữ Ký hiệu Việt Nam (NNKH).
Chuỗi ký hiệu gốc được nhận diện từ camera:
"${inputChain}"

YÊU CẦU BẮT BUỘC:
1. Chuyển chuỗi ký hiệu rời rạc trên thành 1 câu tiếng Việt chuẩn, tự nhiên, có dấu câu phù hợp (ví dụ "TÔI BÂY_GIỜ CẢM_THẤY VUI_VẺ" -> "Tôi bây giờ cảm thấy vui vẻ.").
2. RÀNG BUỘC AN TOÀN TUYỆT ĐỐI: CHỈ được thêm dấu câu hoặc từ nối ngữ pháp tối thiểu. TUYỆT ĐỐI KHÔNG được thêm bất kỳ thông tin, chi tiết, hay từ ngữ nào mang ý mới không có trong chuỗi gốc.
3. Không thêm lời chào, không kèm giải thích, chỉ trả về đúng 1 câu văn bản kết quả.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        temperature: 0.1,
      },
    });

    const smoothedText = response.text ? response.text.trim().replace(/^["']|["']$/g, '') : inputChain;

    return res.json({
      smoothedText,
      source: 'gemini',
    });
  } catch (err: any) {
    console.error('Error smoothing sentence with Gemini:', err);
    // Return friendly fallback
    const { rawText, tokens } = req.body;
    const fallbackText = rawText || (Array.isArray(tokens) ? tokens.join(' ') : '');
    const cleanFallback = fallbackText
      .toLowerCase()
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    const formatted = cleanFallback ? cleanFallback.charAt(0).toUpperCase() + cleanFallback.slice(1) + '.' : '';

    return res.json({
      smoothedText: formatted || fallbackText,
      source: 'fallback',
      warning: 'Không thể kết nối Gemini API, sử dụng bộ làm mượt ngữ pháp cơ bản.',
    });
  }
});

// Setup Vite or static serving
async function startServer() {
  const isProduction = process.env.NODE_ENV === 'production';

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Voice2Hand server running on port ${PORT}`);
  });
}

startServer();
