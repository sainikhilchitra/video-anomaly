"use client";

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area } from "recharts";
import { Activity } from "lucide-react";

interface AnomalyChartProps {
  data: { time: string; score: number }[];
  threshold: number;
}

export function AnomalyChart({ data, threshold }: AnomalyChartProps) {
  return (
    <div className="glass-card p-4 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4 text-cyan-400 font-mono tracking-wider">
        <Activity className="w-4 h-4" />
        <span className="text-xs uppercase">Anomaly pulse detector</span>
      </div>

      <div className="flex-1 w-full min-h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={data[data.length-1]?.score > threshold ? "#ef4444" : "#06b6d4"} stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
            <XAxis dataKey="time" hide />
            <YAxis 
                domain={[0, 1]} 
                ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                stroke="#52525b" 
                fontSize={10} 
                tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
            />
            <Tooltip 
                contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", fontSize: "12px", borderRadius: "8px" }}
                itemStyle={{ color: "#06b6d4" }}
                labelStyle={{ display: "none" }}
            />
            <Area 
                type="monotone" 
                dataKey="score" 
                stroke={data[data.length-1]?.score > threshold ? "#ef4444" : "#06b6d4"}
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorScore)"
                isAnimationActive={false}
            />
            {/* Threshold Line */}
            <Line 
                type="monotone" 
                dataKey={() => threshold} 
                stroke="#ef4444" 
                strokeDasharray="5 5" 
                strokeWidth={1}
                dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 text-[10px] font-mono uppercase">
        <div className="p-2 border border-zinc-800 rounded bg-zinc-900/50">
          <div className="text-zinc-500">Peak_Score</div>
          <div className="text-cyan-400 mt-1">
            {(Math.max(...data.map(d => d.score), 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="p-2 border border-zinc-800 rounded bg-zinc-900/50">
          <div className="text-zinc-500">Status</div>
          <div className={data[data.length-1]?.score > threshold ? "text-red-400 mt-1" : "text-emerald-400 mt-1"}>
            {data[data.length-1]?.score > threshold ? "DETECTED" : "NORMAL"}
          </div>
        </div>
      </div>
    </div>
  );
}
