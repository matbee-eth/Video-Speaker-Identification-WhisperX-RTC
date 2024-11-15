"use client"
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Square, Settings } from 'lucide-react';

interface DetectionSettings {
  vadAggressiveness: number;
  temporalWindow: number;
  minSpeechFrames: number;
}

interface MediaDevice {
  deviceId: string;
  kind: string;
  label: string;
}

interface FaceDetection {
  bbox: [number, number, number, number];
  landmarks: number[][];
  is_speaking: boolean;
  confidence: number;
}

const SpeakerDetectionApp: React.FC = () => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [selectedVideoDevice, setSelectedVideoDevice] = useState<string>('');
  const [selectedAudioDevice, setSelectedAudioDevice] = useState<string>('');
  const [devices, setDevices] = useState<MediaDevice[]>([]);
  const [detections, setDetections] = useState<FaceDetection[]>([]);
  const [detectionSettings, setDetectionSettings] = useState<DetectionSettings>({
    vadAggressiveness: 3,
    temporalWindow: 10,
    minSpeechFrames: 3,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Audio processing configuration
  const SAMPLE_RATE = 16000;
  const FRAME_LENGTH = 480; // 30ms at 16kHz
  const BUFFER_SIZE = 4096;

  const drawDetections = useCallback((detections: FaceDetection[]) => {
    // Draw detections
    requestAnimationFrame(() => {
    detections.forEach((detection) => {
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx) return;
      const [x, y, w, h] = detection.bbox;

      // Draw bounding box
      ctx.strokeStyle = detection.is_speaking ? '#00ff00' : '#ff0000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Draw status text
      ctx.fillStyle = detection.is_speaking ? '#00ff00' : '#ff0000';
      ctx.font = '16px Arial';
      const status = detection.is_speaking
        ? `Speaking (${detection.confidence.toFixed(2)})`
        : 'Not Speaking';
      ctx.fillText(status, x, y - 10);
    });
    });
  }, []);

  const setupWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    wsRef.current = new WebSocket('ws://localhost:8000/ws');

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.results) {
          setDetections(data.results);
          // Force a redraw after receiving new detections
          if (canvasRef.current && videoRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
              ctx.drawImage(videoRef.current, 0, 0);
              drawDetections(data.results);
            }
          }
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
  
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, [drawDetections]);

  const setupAudioProcessing = useCallback((stream: MediaStream) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }

    const audioContext = audioContextRef.current;
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create script processor for raw audio data
    const processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
    processorRef.current = processor;

    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const inputData = e.inputBuffer.getChannelData(0);
      
      // Convert float32 to 16-bit PCM
      const pcmData = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }

      // Send audio data as base64
      const base64Audio = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
      
      if (canvasRef.current) {
        const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);
        const base64Image = imageData.split(',')[1];

        wsRef.current.send(JSON.stringify({
          frame: base64Image,
          audio: base64Audio,
          settings: detectionSettings
        }));
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
  }, [detectionSettings]);

  const processFrame = useCallback(() => {
    if (!isRecording || !videoRef.current || !canvasRef.current || !wsRef.current) return;
  
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
  
    // Update canvas dimensions if needed
    if (canvasRef.current.width !== videoRef.current.videoWidth ||
        canvasRef.current.height !== videoRef.current.videoHeight) {
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
    }
  
    // Draw video frame
    ctx.drawImage(videoRef.current, 0, 0);
  }, [isRecording, videoRef, canvasRef, wsRef]); // Add missing deps


  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: selectedVideoDevice,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: {
          deviceId: selectedAudioDevice,
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });


      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
          }
        };
        setupWebSocket();
        console.log('WebSocket setup');
        requestAnimationFrame(processFrame);
        console.log('Frame processing started');
      }

      streamRef.current = stream;
      setupWebSocket();
      setupAudioProcessing(stream);
      requestAnimationFrame(processFrame);
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  }, [selectedVideoDevice, selectedAudioDevice, setupWebSocket, setupAudioProcessing, processFrame]);

  const stopRecording = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsRecording(false);
    setDetections([]);
  }, []);

  // Initialize available devices
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices()
      .then(devices => {
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        const audioDevices = devices.filter(device => device.kind === 'audioinput');
        
        setDevices([...videoDevices, ...audioDevices]);
        
        if (videoDevices.length > 0) {
          setSelectedVideoDevice(videoDevices[0].deviceId);
        }
        if (audioDevices.length > 0) {
          setSelectedAudioDevice(audioDevices[0].deviceId);
        }
      });
  }, []);

  useEffect(() => {
    if (!videoRef.current) return;
    videoRef.current.onplaying = () => {
      console.log('Video playing');
      requestAnimationFrame(processFrame);
      setIsRecording(true);
    };
    videoRef.current.onpause = () => {
      console.log('Video paused');
      setIsRecording(false);
    };
    videoRef.current.onended = () => {
      console.log('Video ended');
      setIsRecording(false);
    };
  }, [processFrame]);

  return (
    <div className="flex flex-col min-h-screen bg-gray-100 p-4">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Active Speaker Detection</h1>
      </header>

      <main className="flex-grow flex flex-col items-center gap-6">
        <div className="relative w-full max-w-3xl aspect-video bg-black rounded-lg overflow-hidden">
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-cover"
          />
          <video
            ref={videoRef}
            playsInline
          />
        </div>

        <div className="flex gap-4">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium ${
              isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'
            }`}
          >
            {isRecording ? (
              <>
                <Square size={20} />
                Stop Recording
              </>
            ) : (
              <>
                <Play size={20} />
                Start Recording
              </>
            )}
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300"
          >
            <Settings size={20} />
            Settings
          </button>
        </div>

        {showSettings && (
          <div className="w-full max-w-3xl p-6 bg-white rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4">Settings</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Video Device
                </label>
                <select
                  value={selectedVideoDevice}
                  onChange={(e) => setSelectedVideoDevice(e.target.value)}
                  className="w-full p-2 border rounded-md"
                >
                  {devices
                    .filter(device => device.kind === 'videoinput')
                    .map(device => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${device.deviceId}`}
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Audio Device
                </label>
                <select
                  value={selectedAudioDevice}
                  onChange={(e) => setSelectedAudioDevice(e.target.value)}
                  className="w-full p-2 border rounded-md"
                >
                  {devices
                    .filter(device => device.kind === 'audioinput')
                    .map(device => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Microphone ${device.deviceId}`}
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  VAD Aggressiveness (0-3)
                </label>
                <input
                  type="number"
                  min="0"
                  max="3"
                  value={detectionSettings.vadAggressiveness}
                  onChange={(e) => setDetectionSettings(prev => ({
                    ...prev,
                    vadAggressiveness: parseInt(e.target.value)
                  }))}
                  className="w-full p-2 border rounded-md"
                />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default SpeakerDetectionApp;