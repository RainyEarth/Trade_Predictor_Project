import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

extend: {
  colors: {
    neural: '#4F46E5', // a nice purple for NeuralAditya branding
  }
}


export default config