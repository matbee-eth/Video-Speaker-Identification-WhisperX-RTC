"use client"
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Play, Square, AlertCircle, Loader2, User } from 'lucide-react';

interface DetectionResult {
  bbox: number[];
  is_speaking: boolean;
  speaker_id?: number;
  speaker_confidence?: number;
}

interface SpeakerSegment {
  start: number;
  end: number;
  speaker: number;
  text: string;
}

interface DetectionMessage {
  type: "detection_results";
  timestamp: number;
  face_detections: DetectionResult[];
  speaker_segments: SpeakerSegment[];
}

const WebRTCClient = () => {
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [error, setError] = useState<string>('');
  const [transcripts, setTranscripts] = useState<SpeakerSegment[]>([]);
  const [currentSpeaker, setCurrentSpeaker] = useState<number | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const dataChannelRef = useRef<RTCDataChannel | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);

  const drawDetections = useCallback((detections: DetectionResult[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
  
    // Add these lines to match canvas size to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  
    // Rest of your existing code...
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
      if (detection.speaker_id !== undefined) {
        ctx.fillText(`Speaker ${detection.speaker_id}`, x + 5, y - 25);
        if (detection.speaker_confidence !== undefined) {
          ctx.fillText(`Confidence: ${(detection.speaker_confidence * 100).toFixed(1)}%`, x + 5, y - 8);
        }
      } else {
        ctx.fillText(detection.is_speaking ? 'Speaking' : 'Not Speaking', x + 5, y - 8);
      }
    });

    // Update current speaker info
    const activeSpeaker = detections.find(d => d.is_speaking && d.speaker_id !== undefined);
    if (activeSpeaker?.speaker_id !== undefined) {
      setCurrentSpeaker(activeSpeaker.speaker_id);
      setDetectionConfidence(activeSpeaker.speaker_confidence ? activeSpeaker.speaker_confidence * 100 : 0);
    }
  }, []);

  const handleDataChannel = useCallback((channel: RTCDataChannel) => {
    channel.onmessage = (event) => {
      try {
        console.debug('Received detection message:', event);
        const message = JSON.parse(event.data) as DetectionMessage;
        if (message.type === 'detection_results') {
          // Update face detections
          drawDetections(message.face_detections);

          // Update transcripts
          if (message.speaker_segments.length > 0) {
            setTranscripts(prev => [...prev, ...message.speaker_segments].slice(-50)); // Keep last 50 segments
          }
        }
      } catch (error) {
        console.error('Error processing data channel message:', error);
      }
    };

    channel.onopen = () => {
      console.log('Data channel opened');
    };

    channel.onclose = () => {
      console.log('Data channel closed');
    };

    channel.onerror = (error) => {
      console.error('Data channel error:', error);
      setError('Data channel error');
    };
  }, [drawDetections]);


  const cleanup = useCallback(() => {
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
      localStreamRef.current = null;
    }

    if (dataChannelRef.current) {
      dataChannelRef.current.close();
      dataChannelRef.current = null;
    }

    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject = null;
    }

    setConnectionState('disconnected');
  }, []);


  const setupPeerConnection = useCallback(() => {
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    // Create data channel
    const dataChannel = pc.createDataChannel('detections');
    handleDataChannel(dataChannel);
    dataChannelRef.current = dataChannel;

    pc.ontrack = (event) => {
      if (event.track.kind === 'video' && videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
      }
    };

    pc.ondatachannel = (event) => {
      handleDataChannel(event.channel);
    };

    pc.oniceconnectionstatechange = () => {
      console.log('ICE Connection State:', pc.iceConnectionState);
      if (pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'disconnected') {
        setError('Connection lost. Please try again.');
        cleanup();
      }
    };

    return pc;
  }, [cleanup, handleDataChannel]);

  const startStreaming = useCallback(async () => {
    try {
      setConnectionState('connecting');
      setError('');
  
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      localStreamRef.current = stream;
      
      // Add this line to display local video feed
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
  
      // Create and set up peer connection
      const pc = setupPeerConnection();
      peerConnectionRef.current = pc;

      // Add tracks to peer connection
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });

      // Create and send offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Send offer to server via WebSocket
      const ws = new WebSocket('ws://localhost:8000/rtc');
      
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => {
          ws.send(JSON.stringify({
            type: 'offer',
            sdp: pc.localDescription?.sdp
          }));
          resolve();
        };
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(new Error('WebSocket connection failed'));
        };
        ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
        };
      });

      // Handle answer
      ws.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'answer') {
          await pc.setRemoteDescription(new RTCSessionDescription({
            type: 'answer',
            sdp: message.sdp
          }));
          setConnectionState('connected');
        }
      };

    } catch (err) {
      console.error('Error starting stream:', err);
      setError(err instanceof Error ? err.message : 'Failed to start streaming');
      cleanup();
    }
  }, [cleanup, setupPeerConnection]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="grid grid-cols-3 gap-6">
        {/* Video Feed */}
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>Video Feed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover"
                autoPlay
                playsInline
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
                    <Badge variant="secondary">{detectionConfidence.toFixed(1)}% confidence</Badge>
                  </div>
                </div>
              )}
              {connectionState === 'connecting' && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                  <Loader2 className="h-8 w-8 animate-spin" />
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
            <ScrollArea className="h-[400px]">
              <div className="space-y-4">
                {transcripts.map((segment, i) => (
                  <div
                    key={i}
                    className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg space-y-2"
                  >
                    <div className="flex items-center space-x-2">
                      <Badge
                        style={{
                          backgroundColor: `hsl(${segment.speaker * 40}, 70%, 50%)`
                        }}
                      >
                        Speaker {segment.speaker}
                      </Badge>
                      <span className="text-sm text-gray-500">
                        {new Date(segment.start * 1000).toISOString().substr(11, 8)}
                      </span>
                    </div>
                    <p className="text-sm">{segment.text}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-center">
        <Button
          size="lg"
          onClick={connectionState === 'connected' ? cleanup : startStreaming}
          disabled={connectionState === 'connecting'}
        >
          {connectionState === 'connecting' ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Connecting...
            </>
          ) : connectionState === 'connected' ? (
            <>
              <Square className="mr-2 h-4 w-4" />
              Stop Streaming
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" />
              Start Streaming
            </>
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default WebRTCClient;