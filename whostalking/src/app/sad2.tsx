"use client";
import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { 
  Play, 
  Square, 
  Settings, 
  Volume2, 
  VolumeX,
  Mic, 
  MicOff,
  AlertCircle,
  ChevronDown,
  User
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
declare global {
    interface Window {
      webkitAudioContext: typeof AudioContext;
    }
  }
// Device type
interface MediaDeviceInfo {
    deviceId: string;
    groupId: string;
    kind: string;
    label: string;
  }
  
  // Transcript type
  interface Transcript {
    speaker: number;
    text: string;
    timestamp: number;
  }
  
  // Settings type
  interface Settings {
    hfToken: string;
    vadSensitivity: number;
    minSpeakers: number;
    maxSpeakers: number;
    selectedCamera: string;
    selectedMic: string;
    enableNoiseReduction: boolean;
    enableEchoCancellation: boolean;
    quality: 'low' | 'medium' | 'high';
  }
  
  // Face Detection type
  interface FaceDetection {
    bbox: number[];
    speaker_id?: number;
    is_speaking: boolean;
    speaker_confidence: number;
  }

  interface WebSocketMessage {
    face_detections?: FaceDetection[];
    speaker_segments?: Array<{
      speaker: number;
      text: string;
    }>;
  }
  

export default function SpeakerDetectionApp() {
  // State Management
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [micEnabled, setMicEnabled] = useState<boolean>(true);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [error, setError] = useState<string>('');
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [currentSpeaker, setCurrentSpeaker] = useState<number | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number>(0);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Settings State
  const [settings, setSettings] = useState<Settings>({
    hfToken: '',
    vadSensitivity: 3,
    minSpeakers: 1,
    maxSpeakers: 4,
    selectedCamera: '',
    selectedMic: '',
    enableNoiseReduction: true,
    enableEchoCancellation: true,
    quality: 'high'
  });

  // Constants
//   const SAMPLE_RATE = 16000;
//   const FRAME_LENGTH = 480;
  const BUFFER_SIZE = 4096;

  const qualityPresets = useMemo(() => ({
    low: { width: 640, height: 480, frameRate: 15 },
    medium: { width: 1280, height: 720, frameRate: 24 },
    high: { width: 1920, height: 1080, frameRate: 30 }
  }), []);


  // Audio Processing Setup
  const setupAudioProcessing = useCallback((stream: MediaStream) => {
    if (!audioContextRef.current) {
        // Remove the sampleRate parameter
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
  
      const audioContext = audioContextRef.current;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
      processorRef.current = processor;
      
    let audioBuffer: Float32Array[] = [];
  
    processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !micEnabled) return;

      const inputData = e.inputBuffer.getChannelData(0);
      const pcmData = new Int16Array(inputData.length);
      
      for (let i = 0; i < inputData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
      }

      audioBuffer.push(inputData);

      if (audioBuffer.length >= 10) {
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
        
        if (canvasRef.current && videoRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          if (ctx) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
            ctx.drawImage(videoRef.current, 0, 0);
            const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);
            const base64Image = imageData.split(',')[1];

            wsRef.current.send(JSON.stringify({
              timestamp: audioContext.currentTime,
              frame: base64Image,
              audio: base64Audio
            }));
          }
        }
        
        audioBuffer = [];
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
  }, [micEnabled]);
  
  // Drawing Functions
  const drawDetections = useCallback((detections: FaceDetection[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((detection) => {
      const [x, y, w, h] = detection.bbox;
      const color = detection.speaker_id ? 
        `hsl(${detection.speaker_id * 40}, 70%, 50%)` : 
        (detection.is_speaking ? '#00ff00' : '#ff0000');

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Draw label background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 45, w, 40);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      if (detection.speaker_id) {
        ctx.fillText(`Speaker ${detection.speaker_id}`, x + 5, y - 25);
        ctx.fillText(`Confidence: ${(detection.speaker_confidence * 100).toFixed(1)}%`, x + 5, y - 8);
      } else {
        ctx.fillText(detection.is_speaking ? 'Speaking' : 'Not Speaking', x + 5, y - 8);
      }
    });
  }, []);

  // Initialize WebSocket connection
  const initializeWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    wsRef.current = new WebSocket('ws://localhost:8000/ws');

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setError('');
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WebSocketMessage;
        if (data.face_detections) {
          drawDetections(data.face_detections);
          const activeSpeaker = data.face_detections.find(face => face.is_speaking);
          if (activeSpeaker?.speaker_id !== undefined) {
            setCurrentSpeaker(activeSpeaker.speaker_id);
            setDetectionConfidence(Math.round(activeSpeaker.speaker_confidence * 100));
          }
        }
        if (data.speaker_segments !== undefined && data.speaker_segments.length > 0) {
            setTranscripts(prev => [...prev, {
            speaker: (data.speaker_segments ?? [])[0].speaker,
            text: (data.speaker_segments ?? [])[0].text,
            timestamp: Date.now()
          }].slice(-50)); // Keep last 50 transcripts
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Please try again.');
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket closed');
    };
  }, [drawDetections]);
  // Recording Control
  const handleStartRecording = useCallback(async () => {
    try {
      const quality = qualityPresets[settings.quality];
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: settings.selectedCamera,
          ...quality
        },
        audio: {
          deviceId: settings.selectedMic,
          echoCancellation: settings.enableEchoCancellation,
          noiseSuppression: settings.enableNoiseReduction,
          // Remove the sampleRate constraint
          channelCount: 1
        }
      });
  
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      streamRef.current = stream;
      initializeWebSocket();
      setupAudioProcessing(stream);
      setIsRecording(true);
      setError('');
    } catch (err: unknown) {
    console.error('Error starting recording:', err);
    setError(err instanceof Error ? err.message : 'Failed to start recording');
    }
  }, [initializeWebSocket, qualityPresets, settings, setupAudioProcessing]);

  const handleStopRecording = useCallback(() => {
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
    setCurrentSpeaker(null);
    setDetectionConfidence(0);
  }, []);

  // Device Enumeration
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices()
      .then(deviceList => {
        setDevices(deviceList);
        
        const defaultCamera = deviceList.find(d => d.kind === 'videoinput');
        const defaultMic = deviceList.find(d => d.kind === 'audioinput');
        
        setSettings(prev => ({
          ...prev,
          selectedCamera: defaultCamera?.deviceId || '',
          selectedMic: defaultMic?.deviceId || ''
        }));
      })
      .catch((err: unknown) => {
        console.error('Error enumerating devices:', err);
        setError(err instanceof Error ? err.message : 'Failed to access media devices');
      });
  }, []);
  // Cleanup
  useEffect(() => {
    return () => {
      handleStopRecording();
    };
  }, [handleStopRecording]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white p-6">
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Active Speaker Detection</h1>
          <p className="text-gray-400">Real-time speaker identification and transcription</p>
        </div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Detection Settings</DialogTitle>
              <DialogDescription>Configure your detection preferences and devices</DialogDescription>
            </DialogHeader>
            <Tabs defaultValue="devices" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="devices">Devices</TabsTrigger>
                <TabsTrigger value="audio">Audio</TabsTrigger>
                <TabsTrigger value="detection">Detection</TabsTrigger>
              </TabsList>

              <TabsContent value="devices" className="space-y-4">
                <div className="space-y-4">
                  {/* Camera Selection */}
                  <div className="space-y-2">
                    <Label>Camera</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between">
                          {devices.find(d => d.deviceId === settings.selectedCamera)?.label || 'Select Camera'}
                          <ChevronDown className="h-4 w-4 opacity-50" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-full">
                        {devices
                          .filter(d => d.kind === 'videoinput')
                          .map(device => (
                            <DropdownMenuItem
                              key={device.deviceId}
                              onSelect={() => setSettings(s => ({...s, selectedCamera: device.deviceId}))}
                            >
                              {device.label}
                            </DropdownMenuItem>
                          ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Microphone Selection */}
                  <div className="space-y-2">
                    <Label>Microphone</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between">
                          {devices.find(d => d.deviceId === settings.selectedMic)?.label || 'Select Microphone'}
                          <ChevronDown className="h-4 w-4 opacity-50" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-full">
                        {devices
                          .filter(d => d.kind === 'audioinput')
                          .map(device => (
                            <DropdownMenuItem
                              key={device.deviceId}
                              onSelect={() => setSettings(s => ({...s, selectedMic: device.deviceId}))}
                            >
                              {device.label}
                            </DropdownMenuItem>
                          ))}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Quality Selection */}
                  <div className="space-y-2">
                    <Label>Video Quality</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between">
                          {settings.quality.charAt(0).toUpperCase() + settings.quality.slice(1)}
                          <ChevronDown className="h-4 w-4 opacity-50" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent>
                        <DropdownMenuItem onSelect={() => setSettings(s => ({...s, quality: 'low'}))}>
                          Low (640x480)
                        </DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => setSettings(s => ({...s, quality: 'medium'}))}>
                          Medium (720p)
                        </DropdownMenuItem>
                        <DropdownMenuItem onSelect={() => setSettings(s => ({...s, quality: 'high'}))}>
                          High (1080p)
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="audio" className="space-y-4">
                <div className="space-y-4">
                  {/* Audio Settings */}
                  <div className="flex items-center justify-between">
                    <Label>Noise Reduction</Label>
                    <Switch
                      checked={settings.enableNoiseReduction}
                      onCheckedChange={(checked) => 
                        setSettings(s => ({...s, enableNoiseReduction: checked}))
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label>Echo Cancellation</Label>
                    <Switch
                      checked={settings.enableEchoCancellation}
                      onCheckedChange={(checked) => 
                        setSettings(s => ({...s, enableEchoCancellation: checked}))
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>VAD Sensitivity</Label>
                    <Slider
                      value={[settings.vadSensitivity]}
                      min={0}
                      max={3}
                      step={1}
                      onValueChange={([value]) => 
                        setSettings(s => ({...s, vadSensitivity: value}))
                      }
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="detection" className="space-y-4">
                <div className="space-y-4">
                  {/* Detection Settings */}
                  <div className="space-y-2">
                    <Label>Hugging Face Token</Label>
                    <Input
                      type="password"
                      value={settings.hfToken}
                      onChange={(e) => setSettings(s => ({...s, hfToken: e.target.value}))}
                      placeholder="Enter your token"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Min Speakers</Label>
                      <Input
                        type="number"
                        min={1}
                        value={settings.minSpeakers}
                        onChange={(e) => setSettings(s => ({...s, minSpeakers: parseInt(e.target.value)}))}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Speakers</Label>
                      <Input
                        type="number"
                        min={1}
                        value={settings.maxSpeakers}
                        onChange={(e) => setSettings(s => ({...s, maxSpeakers: parseInt(e.target.value)}))}
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </DialogContent>
        </Dialog>
      </header>

      {/* Main Content */}
      <div className="grid grid-cols-3 gap-6">
        {/* Video Feed */}
        <Card className="col-span-2">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Video Feed</CardTitle>
            <div className="flex space-x-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMicEnabled(!micEnabled)}
              >
                {micEnabled ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsMuted(!isMuted)}
              >
                {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover"
                playsInline
                muted={isMuted}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full"
              />
              {currentSpeaker && (
                <div className="absolute bottom-4 left-4 right-4 bg-black/50 backdrop-blur-sm rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <User className="h-5 w-5" />
                    <span className="font-medium">Speaker {currentSpeaker}</span>
                    <Badge variant="secondary">{detectionConfidence}% confidence</Badge>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Transcripts */}
        <Card>
          <CardHeader>
            <CardTitle>Transcripts</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px] pr-4">
              <div className="space-y-4">
                {transcripts.map((transcript, i) => (
                  <div
                    key={i}
                    className="p-3 bg-gray-800 rounded-lg space-y-2"
                  >
                    <div className="flex items-center space-x-2">
                      <Badge
                        style={{
                          backgroundColor: `hsl(${transcript.speaker * 40}, 70%, 50%)`
                        }}
                      >
                        Speaker {transcript.speaker}
                      </Badge>
                      <span className="text-sm text-gray-400">
                        {new Date(transcript.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm">{transcript.text}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Controls */}
      <div className="flex justify-center">
        <Button
          size="lg"
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          className={isRecording ? 'bg-red-500 hover:bg-red-600' : ''}
        >
          {isRecording ? (
            <>
              <Square className="mr-2 h-4 w-4" />
              Stop Recording
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Start Recording
            </>
          )}
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  </div>
  );
}