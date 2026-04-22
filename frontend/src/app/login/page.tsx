'use client'

import { useState } from 'react'
import { login, signup, verifyOtp, resendVerification } from './actions'
import { Shield, Mail, Lock, Loader2, AlertCircle, KeyRound, ArrowRight, Send } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [confirmPassword, setConfirmPassword] = useState('')
  const [validationError, setValidationError] = useState('')

  const searchParams = useSearchParams()
  const error = searchParams.get('error')
  const message = searchParams.get('message')
  const isVerify = searchParams.get('verify') === 'true'
  const emailFromUrl = searchParams.get('email') || ''
  const verificationType = searchParams.get('type') || 'signup'

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    setValidationError('')
    
    if (!isLogin && !isVerify) {
      const formData = new FormData(event.currentTarget)
      const password = formData.get('password') as string
      if (password !== confirmPassword) {
        event.preventDefault()
        setValidationError('Access ciphers do not match.')
        return
      }
    }
    
    setIsLoading(true)
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-[#09090b] relative overflow-hidden">
      {/* Background Decor */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[100px] animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[100px] animate-pulse delay-1000" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md z-10"
      >
        <div className="glass-card p-8 glow-cyan relative">
          <div className="scanline" />
          
          <div className="flex flex-col items-center mb-8">
            <div className="p-3 bg-cyan-500/10 rounded-xl border border-cyan-500/30 mb-4">
              <Shield className="w-8 h-8 text-cyan-400" />
            </div>
            <h1 className="text-2xl font-bold tracking-tighter uppercase font-mono text-white text-center">
              {isVerify ? 'Verification Required' : isLogin ? 'Secure Access' : 'Register Operator'}
            </h1>
            <p className="text-xs font-mono text-cyan-500/60 uppercase tracking-widest mt-2">
              {isVerify ? 'Proof of Identity' : 'Temporal Attention System v4.0'}
            </p>
          </div>

          <AnimatePresence mode="wait">
            {(error || validationError) && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex flex-col gap-2 text-red-400 text-xs font-mono"
              >
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{error || validationError}</span>
                </div>
                {error?.toLowerCase().includes('verify') && (
                  <form action={resendVerification} className="mt-1">
                    <input type="hidden" name="email" value={emailFromUrl} />
                    <button type="submit" className="text-[10px] text-red-400/70 hover:text-red-400 underline underline-offset-4 flex items-center gap-1">
                      <Send className="w-3 h-3" /> Resend code
                    </button>
                  </form>
                )}
              </motion.div>
            )}
            {message && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6 p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg flex items-center gap-3 text-cyan-400 text-xs font-mono"
              >
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{message}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <form action={isVerify ? verifyOtp : isLogin ? login : signup} onSubmit={handleSubmit} className="space-y-6">
            <input type="hidden" name="type" value={verificationType} />
            <div className="space-y-2">
              <label className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest block ml-1">
                Operator ID (Email)
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600" />
                <input
                  name="email"
                  type="email"
                  required
                  defaultValue={emailFromUrl}
                  readOnly={isVerify}
                  placeholder="operator@system.io"
                  className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2.5 pl-10 pr-4 text-sm font-mono text-white placeholder:text-zinc-700 focus:outline-none focus:border-cyan-500/50 transition-all disabled:opacity-50"
                />
              </div>
            </div>

            {!isVerify && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest block ml-1">
                    Access Cipher (Password)
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600" />
                    <input
                      name="password"
                      type="password"
                      required
                      placeholder="••••••••"
                      className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2.5 pl-10 pr-4 text-sm font-mono text-white placeholder:text-zinc-700 focus:outline-none focus:border-cyan-500/50 transition-all"
                    />
                  </div>
                </div>

                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-2 overflow-hidden"
                    >
                      <label className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest block ml-1">
                        Confirm Access Cipher
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-600" />
                        <input
                          name="confirmPassword"
                          type="password"
                          required
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                          placeholder="••••••••"
                          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2.5 pl-10 pr-4 text-sm font-mono text-white placeholder:text-zinc-700 focus:outline-none focus:border-cyan-500/50 transition-all"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}

            {isVerify && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-2"
              >
                <label className="text-[10px] font-mono text-emerald-500 uppercase tracking-widest block ml-1">
                  8-Digit Verification Cipher
                </label>
                <div className="relative">
                  <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-emerald-600" />
                  <input
                    name="token"
                    type="text"
                    required
                    autoFocus
                    placeholder="00000000"
                    maxLength={8}
                    className="w-full bg-zinc-950 border border-emerald-500/30 rounded-lg py-2.5 pl-10 pr-4 text-lg tracking-[0.3em] font-mono text-white placeholder:text-zinc-800 focus:outline-none focus:border-emerald-500/50 transition-all text-center"
                  />
                </div>
              </motion.div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className={`w-full py-3 rounded-lg text-xs font-mono uppercase transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:pointer-events-none ${
                isVerify || !isLogin 
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20'
                  : 'bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20'
              } active:scale-[0.98]`}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <>
                  {isVerify ? 'Confirm Registration' : isLogin ? 'Initiate Link' : 'Create Operator Account'}
                  <ArrowRight className="w-3 h-3" />
                </>
              )}
            </button>
          </form>

          {!isVerify && (
            <div className="mt-8 pt-6 border-t border-zinc-800 flex flex-col items-center gap-4">
              <button
                onClick={() => {
                  setIsLogin(!isLogin)
                }}
                className="text-[10px] font-mono text-zinc-500 uppercase hover:text-cyan-400 transition-colors"
              >
                {isLogin ? "Request New Operator Credentials" : "Existing Operator? Log In"}
              </button>
            </div>
          )}

          {isVerify && (
            <div className="mt-8 pt-6 border-t border-zinc-800 flex flex-col items-center gap-4">
              <Link
                href="/login"
                className="text-[10px] font-mono text-zinc-500 uppercase hover:text-cyan-400 transition-colors"
              >
                Cancel & Back to login
              </Link>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}
