import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': {
        target: 'http://127.0.0.1:30001',
        changeOrigin: true,
      },
      '/queries': {
        target: 'http://127.0.0.1:30001',
        changeOrigin: true,
      },
      '/search': {
        target: 'http://127.0.0.1:30001',
        changeOrigin: true,
      },
      '/advice': {
        target: 'http://127.0.0.1:30001',
        changeOrigin: true,
      }
    },
    allowedHosts: ["clicchat.titusng.cc"]
  }
})
