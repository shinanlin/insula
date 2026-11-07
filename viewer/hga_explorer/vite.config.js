import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const anatomyPort = process.env.ANATOMY_API_PORT || '18082';
const anatomyProxy = {
  '/api': {
    target: `http://127.0.0.1:${anatomyPort}`,
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: anatomyProxy,
  },
  preview: {
    proxy: anatomyProxy,
  },
});
