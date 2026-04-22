"use client";

import { useState, useCallback, useEffect } from "react";
import { VideoFeed } from "@/components/CommandCenter/VideoFeed";
import { AnomalyChart } from "@/components/CommandCenter/AnomalyChart";
import { IncidentLog } from "@/components/CommandCenter/IncidentLog";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Shield, LogOut, Activity, History, Settings, Bell, User, Monitor, Video, AlertTriangle, CheckCircle2, TrendingUp } from "lucide-react";
import { createClient } from "@/utils/supabase/client";
import { useRouter } from "next/navigation";
import { logout } from "./login/actions";
import { HistoricalAnalytics } from "@/components/CommandCenter/HistoricalAnalytics";

interface Incident {
  id: string;
  time: string;
  score: number;
  description: string;
}

export default function Home() {
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
  const { isReady, val, send, status } = useWebSocket(wsUrl);
  
  const [sourceType, setSourceType] = useState<"webcam" | "video">("webcam");
  const [anomalyData, setAnomalyData] = useState<{ time: string; score: number }[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [historyRecords, setHistoryRecords] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<"live" | "history">("live");
  const router = useRouter();
  const supabase = createClient();
  const [user, setUser] = useState<any>(null);
  const [threshold, setThreshold] = useState(0.8);
  const [lastAnomalyTime, setLastAnomalyTime] = useState(0);

  const peakScore = Math.max(...anomalyData.map(d => d.score), 0);

  useEffect(() => {
    const init = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
      if (user) {
        const { data, error } = await supabase.from('anomaly_records').select('*').order('created_at', { ascending: false }).limit(50);
        if (data && !error) {
          const records = data.map((d: any) => ({
            id: d.id,
            time: new Date(d.created_at).toLocaleTimeString(),
            score: d.anomaly_score,
            source: d.video_source,
            status: d.status || 'pending',
            description: d.additional_data?.description || "Behavioral Anomaly"
          }));
          setHistoryRecords(records);
          setIncidents(records.slice(0, 10));
        }
      }
    };
    init();
  }, [supabase]);

  const handleUpdateStatus = async (id: string, status: 'verified' | 'false_alarm') => {
    const { error } = await supabase.from('anomaly_records').update({ status }).eq('id', id);
    if (!error) setHistoryRecords(prev => prev.map(r => r.id === id ? { ...r, status } : r));
  };

  const handleLogout = async () => {
    await logout();
  };

  useEffect(() => {
    if (val && val.score !== undefined) {
      const newPoint = {
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        score: val.score
      };
      setAnomalyData(prev => [...prev.slice(-49), newPoint]);
      if (val.score > threshold && Date.now() - lastAnomalyTime > 5000) {
        const newIncident: Incident = { id: Math.random().toString(36).substr(2, 9), time: new Date().toLocaleTimeString(), score: val.score, description: "Potential Behavioral Anomaly" };
        setIncidents(prev => [newIncident, ...prev.slice(0, 19)]);
        setLastAnomalyTime(Date.now());
        if (user) {
          supabase.from('anomaly_records').insert({ user_id: user.id, video_source: sourceType, anomaly_score: val.score, is_anomaly: true, additional_data: { description: newIncident.description } }).select().single().then(({ data, error }) => {
            if (data && !error) {
              setHistoryRecords(prev => [{ id: data.id, time: new Date(data.created_at).toLocaleTimeString(), score: data.anomaly_score, source: data.video_source, status: data.status || 'pending', description: data.additional_data?.description || "Behavioral Anomaly" }, ...prev.slice(0, 49)]);
            }
          });
        }
      }
    }
  }, [val, threshold, lastAnomalyTime, user, sourceType, supabase]);

  const handleFrame = useCallback((base64: string) => {
    if (isReady) send({ image: base64, timestamp: Date.now(), threshold: threshold });
  }, [isReady, send, threshold]);

  return (
    <main className="min-h-screen bg-[#09090b] text-zinc-300 p-4 md:p-6 lg:p-8 flex flex-col gap-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <nav className="flex items-center justify-between glass-card p-4 glow-cyan">
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-cyan-400" />
          <div>
            <h1 className="text-xl font-bold tracking-tighter uppercase text-white">Video Anomaly Detection</h1>
            <p className="text-[10px] text-zinc-500 uppercase">Secure Operator Terminal</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 pr-4 border-r border-zinc-800">
             <button onClick={() => setSourceType("webcam")} className={`px-3 py-1 rounded text-xs transition-all ${sourceType === "webcam" ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500"}`}>Webcam</button>
             <button onClick={() => setSourceType("video")} className={`px-3 py-1 rounded text-xs transition-all ${sourceType === "video" ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500"}`}>File</button>
          </div>
          <button onClick={handleLogout} className="flex items-center gap-2 px-3 py-1.5 rounded bg-red-500/10 text-red-400 text-xs hover:bg-red-500/20 transition-all">
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </div>
      </nav>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 flex items-center justify-between">
          <div>
            <p className="text-[10px] text-zinc-500 uppercase">System Status</p>
            <p className="text-sm font-bold text-white mt-1 uppercase">{status}</p>
          </div>
          <Activity className={`w-5 h-5 ${status === 'Connected' ? 'text-cyan-500' : 'text-red-500'}`} />
        </div>
        <div className="glass-card p-4 flex items-center justify-between">
          <div>
            <p className="text-[10px] text-zinc-500 uppercase">Peak Score</p>
            <p className="text-sm font-bold text-white mt-1">{(peakScore * 100).toFixed(1)}%</p>
          </div>
          <TrendingUp className="w-5 h-5 text-zinc-500" />
        </div>
        <div className="glass-card p-4 flex items-center justify-between">
          <div>
            <p className="text-[10px] text-zinc-500 uppercase">Records</p>
            <p className="text-sm font-bold text-white mt-1">{historyRecords.length}</p>
          </div>
          <History className="w-5 h-5 text-zinc-500" />
        </div>
        <div className="glass-card p-4 flex items-center justify-between">
          <div>
            <p className="text-[10px] text-zinc-500 uppercase">Operator Rank</p>
            <p className="text-sm font-bold text-cyan-400 mt-1 uppercase">{historyRecords.filter(r => r.status === 'verified').length > 5 ? 'Senior' : 'Recruit'}</p>
          </div>
          <Shield className="w-5 h-5 text-cyan-500" />
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 space-y-6">
          <VideoFeed onFrame={handleFrame} isAnomaly={val?.is_anomaly ?? false} prediction={val?.prediction} status={status} sourceType={sourceType} />
          <AnomalyChart data={anomalyData} threshold={threshold} />
        </div>

        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="glass-card flex flex-col h-[350px] overflow-hidden">
             <div className="flex border-b border-zinc-800 shrink-0">
                <button onClick={() => setActiveTab("live")} className={`flex-1 py-3 text-xs uppercase tracking-widest ${activeTab === "live" ? "bg-cyan-500/10 text-cyan-400" : "text-zinc-500"}`}>Live Feed</button>
                <button onClick={() => setActiveTab("history")} className={`flex-1 py-3 text-xs uppercase tracking-widest ${activeTab === "history" ? "bg-cyan-500/10 text-cyan-400" : "text-zinc-500"}`}>Archive</button>
             </div>
             <div className="flex-1 relative">
                <div className="absolute inset-0 p-4">
                   {activeTab === "live" ? <IncidentLog incidents={incidents} /> : <HistoricalAnalytics records={historyRecords} onUpdateStatus={handleUpdateStatus} />}
                </div>
             </div>
          </div>

          <div className="glass-card p-6 space-y-4">
             <div className="flex justify-between text-xs text-zinc-500 uppercase">
                <span>Sensitivity</span>
                <span>{(threshold * 100).toFixed(0)}%</span>
             </div>
             <input type="range" min="0.1" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} className="w-full h-2 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-cyan-500" />
             <button onClick={() => {setAnomalyData([]); setIncidents([]);}} className="w-full py-2 border border-zinc-800 rounded text-xs uppercase text-zinc-500 hover:bg-zinc-800 transition-all">Clear Session</button>
          </div>
        </div>
      </div>
    </main>
  );
}
