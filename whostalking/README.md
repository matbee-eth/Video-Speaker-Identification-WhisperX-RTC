# Who's Talking - Real-Time Speaker Detection UI

This is the frontend application for the Creative Speaker Detection system, built with [Next.js](https://nextjs.org). It provides a real-time interface for speaker detection and transcription using WebRTC.

## Features

- Real-time video streaming with WebRTC
- Live speaker detection visualization
- Face tracking with bounding boxes
- Dynamic speaker identification
- Real-time transcription display
- Confidence scoring for speaker detection
- Responsive design with dark mode support

## Getting Started

First, ensure the Creative Speaker Detection backend server is running. Then start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```
Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Configuration
The application expects the WebRTC backend server to be running at `ws://localhost:8000/rtc`. You can modify this in `src/app/webrtc.tsx` if needed.

## Technology Stack
- Next.js 13+ with App Router
- React 18
- TypeScript
- WebRTC for real-time communication
- Tailwind CSS for styling
- Radix UI components
- Lucide React icons

## Project Structure
- `src/app/webrtc.tsx` - Main WebRTC client component
- `src/components/ui/` - Reusable UI components
- `public/` - Static assets

## Development
You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses `next/font` to automatically optimize and load Geist, a custom font family.

## Learn More
To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## Deployment
The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/deployment) for more details.