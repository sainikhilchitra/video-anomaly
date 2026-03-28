"use client";

import { AlertTriangle, Clock, MapPin } from "lucide-react";
import { cn } from "@/lib/utils";

interface Incident {
    id: string;
    time: string;
    score: number;
    description: string;
}

interface IncidentLogProps {
    incidents: Incident[];
}

export function IncidentLog({ incidents }: IncidentLogProps) {
    return (
        <div className="glass-card p-4 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-cyan-400 font-mono tracking-wider">
                    <Clock className="w-4 h-4" />
                    <span className="text-xs uppercase">Recent Incidents</span>
                </div>
                <div className="px-2 py-0.5 rounded bg-zinc-800 text-[10px] text-zinc-400 border border-zinc-700">
                    {incidents.length} TOTAL
                </div>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-zinc-800">
                {incidents.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center opacity-30 text-center py-10">
                        <MapPin className="w-8 h-8 mb-2" />
                        <span className="text-xs font-mono uppercase tracking-widest">No Sector Anomalies</span>
                    </div>
                ) : (
                    incidents.map((incident) => (
                        <div key={incident.id} className="p-3 border border-red-950/30 bg-red-950/10 rounded-lg group hover:border-red-500/30 transition-colors">
                            <div className="flex items-start justify-between">
                                <div className="flex gap-3">
                                    <div className="p-2 bg-red-500/10 rounded border border-red-500/20 text-red-500">
                                        <AlertTriangle className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <div className="text-xs font-mono text-white mb-0.5">{incident.description}</div>
                                        <div className="text-[10px] font-mono text-zinc-500">{incident.time}</div>
                                    </div>
                                </div>
                                <div className="text-[10px] font-mono font-bold text-red-500 tabular-nums">
                                    {(incident.score * 100).toFixed(0)}%
                                </div>
                            </div>
                        </div>
                    ))
                ).reverse()}
            </div>
            
            <div className="mt-4 pt-4 border-t border-zinc-800 text-[9px] font-mono text-zinc-600 uppercase flex justify-between">
                <span>Database: secure_vad_v1</span>
                <span className="animate-pulse">Live_Sync_Active</span>
            </div>
        </div>
    );
}
