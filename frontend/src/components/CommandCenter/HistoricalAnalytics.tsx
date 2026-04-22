'use client'

import { useState } from 'react'
import { Calendar, Filter, CheckCircle2, XCircle, AlertTriangle, Clock } from 'lucide-react'
import { motion } from 'framer-motion'

interface HistoryRecord {
  id: string
  time: string
  score: number
  source: string
  status: 'pending' | 'verified' | 'false_alarm'
  description: string
}

interface HistoricalAnalyticsProps {
  records: HistoryRecord[]
  onUpdateStatus: (id: string, status: 'verified' | 'false_alarm') => void
}

export function HistoricalAnalytics({ records, onUpdateStatus }: HistoricalAnalyticsProps) {
  const [filter, setFilter] = useState<'all' | 'pending' | 'verified' | 'false_alarm'>('all')

  const filteredRecords = records.filter(r => filter === 'all' || r.status === filter)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2 text-cyan-400 font-mono text-[10px] uppercase tracking-widest">
          <Calendar className="w-4 h-4" />
          <span>Archive</span>
        </div>
        
        <div className="flex gap-1.5 bg-zinc-900/50 p-1 rounded border border-zinc-800">
          {(['all', 'pending', 'verified', 'false_alarm'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-0.5 rounded text-[8px] font-mono uppercase transition-all ${
                filter === f ? 'bg-cyan-500/20 text-cyan-400' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
        {filteredRecords.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-zinc-600 font-mono gap-2 opacity-30 py-10">
             <Filter className="w-8 h-8" />
             <span className="text-[10px] uppercase">No records</span>
          </div>
        ) : (
          filteredRecords.map((record, index) => (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              key={record.id}
              className="p-3 border border-zinc-800 bg-zinc-900/20 rounded-lg hover:border-cyan-500/30 transition-all group"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded border ${
                    record.status === 'verified' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                    record.status === 'false_alarm' ? 'bg-red-500/10 border-red-500/30 text-red-400' :
                    'bg-amber-500/10 border-amber-500/30 text-amber-400'
                  }`}>
                    {record.status === 'verified' ? <CheckCircle2 className="w-3.5 h-3.5" /> :
                     record.status === 'false_alarm' ? <XCircle className="w-3.5 h-3.5" /> :
                     <AlertTriangle className="w-3.5 h-3.5" />}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                       <span className="text-[10px] font-bold text-white font-mono uppercase tracking-tighter">{record.description}</span>
                       <span className="text-[8px] bg-zinc-800 text-zinc-600 px-1 py-0.5 rounded font-mono uppercase">{record.source}</span>
                    </div>
                    <div className="flex items-center gap-3 mt-1">
                       <div className="flex items-center gap-1 text-[9px] text-zinc-600 font-mono uppercase">
                          <Clock className="w-3 h-3" />
                          {record.time}
                       </div>
                       <div className="text-[9px] font-mono text-cyan-500/40 uppercase">
                          {(record.score * 100).toFixed(0)}%
                       </div>
                    </div>
                  </div>
                </div>

                {record.status === 'pending' && (
                  <div className="flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                      onClick={() => onUpdateStatus(record.id, 'verified')}
                      className="p-1.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded hover:bg-emerald-500/20 transition-all"
                    >
                      <CheckCircle2 className="w-3 h-3" />
                    </button>
                    <button 
                      onClick={() => onUpdateStatus(record.id, 'false_alarm')}
                      className="p-1.5 bg-red-500/10 border border-red-500/30 text-red-400 rounded hover:bg-red-500/20 transition-all"
                    >
                      <XCircle className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  )
}
