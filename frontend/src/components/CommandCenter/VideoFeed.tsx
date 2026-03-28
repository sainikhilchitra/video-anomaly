"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Camera, CameraOff, AlertTriangle, ShieldCheck, Upload, Play, Power } from "lucide-react";
import { cn } from "@/lib/utils";

interface VideoFeedProps {
  onFrame: (base64: string) => void;
  isAnomaly: boolean;
  prediction?: string;
  status: string;
  sourceType: "webcam" | "video";
}

export function VideoFeed({ onFrame, isAnomaly, prediction, status, sourceType }: VideoFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [isActive, setIsActive] = useState(false);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  
  // Manage Webcam stream
  useEffect(() => {
    if (sourceType !== "webcam") {
        setIsActive(false);
        return;
    }

    let stream: MediaStream | null = null;
    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480, frameRate: 15 } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsActive(true);
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    }
    startCamera();
    
    return () => {
      stream?.getTracks().forEach(track => track.stop());
    };
  }, [sourceType]);

  // Handle File Upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
      const url = URL.createObjectURL(file);
      setFileUrl(url);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        videoRef.current.loop = true;
        videoRef.current.play();
        setIsActive(true);
      }
    }
  };

  // Capture loop
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        // Only capture if video is playing and has data
        if (videoRef.current.paused || videoRef.current.readyState < 2) return;

        const ctx = canvasRef.current.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, 128, 128);
          // High quality jpeg to preserve attention cues
          const base64 = canvasRef.current.toDataURL("image/jpeg", 0.85);
          onFrame(base64);
        }
      }
    }, 100); // 10 FPS

    return () => clearInterval(interval);
  }, [isActive, onFrame]);

  return (
    <div className="relative w-full aspect-video glass-card glow-cyan overflow-hidden group">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "w-full h-full object-cover transition-all duration-700",
          isAnomaly ? "grayscale-[0.5] sepia-[0.3] hue-rotate-[320deg] scale-[1.02]" : "grayscale-0"
        )}
      />
      
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} width={128} height={128} className="hidden" />

      {/* Overlays */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="scanline" />
        
        {/* Status Indicators */}
        <div className="absolute top-4 left-4 flex gap-2">
          <div className={cn(
            "px-3 py-1.5 rounded-full flex items-center gap-2 backdrop-blur-md border text-xs font-mono uppercase tracking-wider transition-all duration-300",
            status === "connected" ? "border-cyan-500/50 bg-cyan-950/40 text-cyan-400" : "border-zinc-700 bg-zinc-900/40 text-zinc-400"
          )}>
            <div className={cn("w-2 h-2 rounded-full animate-pulse", status === "connected" ? "bg-cyan-500" : "bg-zinc-500")} />
            Link: {status}
          </div>
          
          <div className={cn(
            "px-3 py-1.5 rounded-full flex items-center gap-2 backdrop-blur-md border text-xs font-mono uppercase tracking-wider transition-all duration-300",
            isAnomaly ? "border-red-500/50 bg-red-950/40 text-red-400 glow-red" : "border-emerald-500/50 bg-emerald-950/40 text-emerald-400"
          )}>
            {isAnomaly ? <AlertTriangle className="w-3 h-3" /> : <ShieldCheck className="w-3 h-3" />}
            SYSTEM: {isAnomaly ? "ANOMALY DETECTED" : "SECURE"}
          </div>
        </div>

        {/* HUD Elements */}
        <div className="absolute bottom-4 right-4 text-[10px] font-mono text-cyan-500/50 uppercase leading-none text-right">
          <div>Cam_ID: {sourceType === 'webcam' ? 'SRC_LIVE' : 'SRC_FILE'}</div>
          <div className="mt-1">SOURCE: {sourceType.toUpperCase()}</div>
          <div className="mt-1">ENC: H.264 / 15 FPS</div>
        </div>

        {/* Prediction Preview Box */}
        {prediction && (
          <div className="absolute bottom-4 left-4 p-2 glass-card border-zinc-700/50 bg-black/40">
             <div className="text-[10px] text-cyan-400 mb-1 font-mono uppercase">Model prediction</div>
             <img src={prediction} alt="Prediction" className="w-32 h-32 opacity-80" />
          </div>
        )}
      </div>

      {/* Source Controls (Clickable) */}
      {sourceType === "video" && !fileUrl && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
           <button 
             onClick={() => fileInputRef.current?.click()}
             className="px-6 py-3 bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 rounded-xl flex items-center gap-3 hover:bg-cyan-500/30 transition-all active:scale-95 pointer-events-auto"
           >
              <Upload className="w-5 h-5" />
              <span className="font-mono uppercase tracking-widest text-sm">Upload Surveillance Clip</span>
           </button>
           <input 
             ref={fileInputRef}
             type="file" 
             accept="video/*" 
             onChange={handleFileChange} 
             className="hidden" 
           />
        </div>
      )}
    </div>
  );
}
