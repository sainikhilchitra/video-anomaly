"use client";

import { useState, useCallback, useEffect } from "react";
import { VideoFeed } from "@/components/CommandCenter/VideoFeed";
import { AnomalyChart } from "@/components/CommandCenter/AnomalyChart";
import { IncidentLog } from "@/components/CommandCenter/IncidentLog";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Shield, LayoutDashboard, Settings, Bell, User, Monitor, Video } from "lucide-react";

interface Incident {
  id: string;
  time: string;
  score: number;
  description: string;
}

export default function Home() {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
  const { isReady, val, send, status } = useWebSocket(wsUrl);
  
  // App State
  const [sourceType, setSourceType] = useState<"webcam" | "video">("webcam");
  const [anomalyData, setAnomalyData] = useState<{ time: string; score: number }[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [threshold, setThreshold] = useState(0.8);
  const [lastAnomalyTime, setLastAnomalyTime] = useState(0);

  // Handle incoming WebSocket data
  useEffect(() => {
    if (val && val.score !== undefined) {
      const newPoint = {
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        score: val.score
      };
      
      setAnomalyData(prev => [...prev.slice(-49), newPoint]);

      // Detect Incident (Cooldown of 5 seconds)
      if (val.score > threshold && Date.now() - lastAnomalyTime > 5000) {
        const newIncident: Incident = {
          id: Math.random().toString(36).substr(2, 9),
          time: new Date().toLocaleTimeString(),
          score: val.score,
          description: "Potential Behavioral Anomaly"
        };
        setIncidents(prev => [...prev, newIncident]);
        setLastAnomalyTime(Date.now());
      }
    }
  }, [val, threshold, lastAnomalyTime]);

  const handleFrame = useCallback((base64: string) => {
    if (isReady) {
      send({ image: base64, timestamp: Date.now(), threshold: threshold });
    }
  }, [isReady, send, threshold]);

  const handleRecalibrate = () => {
    setAnomalyData([]);
    setIncidents([]);
    setLastAnomalyTime(0);
  };

  const handleExportLog = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(incidents, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href",     dataStr);
    downloadAnchorNode.setAttribute("download", `anomaly_log_${new Date().getTime()}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  return (
    <main className="min-h-screen p-4 md:p-6 lg:p-8 flex flex-col gap-6 max-w-[1600px] mx-auto overflow-hidden">
      {/* Navbar */}
      <nav className="flex items-center justify-between glass-card p-4 glow-cyan border-cyan-500/20">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
            <Shield className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tighter uppercase font-mono text-white">Project: AE-VAD v4.0</h1>
            <p className="text-[10px] font-mono text-cyan-500/60 uppercase">Advanced Temporal Attention System</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="hidden md:flex gap-4">
             {/* Source Toggle */}
             <div className="flex bg-zinc-900 border border-zinc-800 rounded-lg p-1">
                <button 
                  onClick={() => setSourceType("webcam")}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-[10px] font-mono uppercase transition-all ${sourceType === "webcam" ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500 hover:text-zinc-300"}`}
                >
                   <Monitor className="w-3 h-3" /> Live Webcam
                </button>
                <button 
                  onClick={() => setSourceType("video")}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-[10px] font-mono uppercase transition-all ${sourceType === "video" ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500 hover:text-zinc-300"}`}
                >
                   <Video className="w-3 h-3" /> Surveillance File
                </button>
             </div>
          </div>

          <div className="flex items-center gap-4">
             <Bell className="w-4 h-4 text-zinc-500 cursor-pointer hover:text-cyan-400 transition-colors" />
             <div className="w-8 h-8 rounded-full border border-zinc-700 bg-zinc-800 flex items-center justify-center">
                <User className="w-4 h-4 text-zinc-400" />
             </div>
          </div>
        </div>
      </nav>

      {/* Main Grid */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0">
        
        {/* Left Column: Feed & Chart */}
        <div className="lg:col-span-8 flex flex-col gap-6 min-h-0">
          <div className="flex-1">
             <VideoFeed 
                onFrame={handleFrame} 
                isAnomaly={val?.is_anomaly ?? false} 
                prediction={val?.prediction}
                status={status}
                sourceType={sourceType}
             />
          </div>
          
          <div className="h-[300px]">
             <AnomalyChart data={anomalyData} threshold={threshold} />
          </div>
        </div>

        {/* Right Column: Sidebar */}
        <div className="lg:col-span-4 flex flex-col gap-6 min-h-0 overflow-hidden">
          <div className="flex-1 min-h-0">
             <IncidentLog incidents={incidents} />
          </div>
          
          {/* Quick Controls Card */}
          <div className="glass-card p-4">
             <div className="flex items-center gap-2 mb-4 text-cyan-400 font-mono tracking-wider">
                <Settings className="w-4 h-4" />
                <span className="text-xs uppercase">System parameters</span>
             </div>
             
             <div className="space-y-4">
                <div>
                   <div className="flex justify-between text-[10px] font-mono text-zinc-500 uppercase mb-2">
                      <span>Sensitivity Gate</span>
                      <span>{(threshold * 100).toFixed(0)}%</span>
                   </div>
                   <input 
                      type="range"
                      min="0.1"
                      max="0.95"
                      step="0.05"
                      value={threshold}
                      onChange={(e) => setThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-cyan-500 hover:accent-cyan-400 transition-all"
                   />
                </div>

                <div className="grid grid-cols-2 gap-2">
                   <button 
                      onClick={handleRecalibrate}
                      className="p-2 border border-cyan-500/30 bg-cyan-500/5 rounded text-[10px] font-mono uppercase text-cyan-400 hover:bg-cyan-500/10 active:scale-95 transition-all"
                   >
                      Recalibrate
                   </button>
                   <button 
                      onClick={handleExportLog}
                      className="p-2 border border-zinc-800 bg-zinc-900 rounded text-[10px] font-mono uppercase text-zinc-500 hover:border-zinc-700 active:scale-95 transition-all"
                   >
                      Export Log
                   </button>
                </div>
             </div>
          </div>
        </div>
      </div>

      {/* Footer Meta */}
      <footer className="mt-auto pt-4 flex items-center justify-between text-[8px] font-mono text-zinc-600 uppercase tracking-widest border-t border-zinc-500/10">
         <div className="flex gap-4">
            <span>Server: {wsUrl.split('://')[1].split('/')[0]}</span>
            <span>Prot: {wsUrl.startsWith('wss') ? 'WSS' : 'WS'}</span>
            <span className="text-cyan-800">Mode: {sourceType.toUpperCase()}</span>
         </div>
         <div>
            Attention-Enhanced VAD Framework © 2026
         </div>
      </footer>
    </main>
  );
}
