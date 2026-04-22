'use server'

import { revalidatePath } from 'next/cache'
import { redirect } from 'next/navigation'
import { createClient } from '@/utils/supabase/server'

export async function login(formData: FormData) {
  const supabase = await createClient()

  const data = {
    email: formData.get('email') as string,
    password: formData.get('password') as string,
  }

  const { data: authData, error } = await supabase.auth.signInWithPassword(data)

  if (error) {
    return redirect('/login?error=' + encodeURIComponent(error.message))
  }

  // Check if email is confirmed
  if (authData.user && !authData.user.email_confirmed_at) {
    await supabase.auth.signOut()
    return redirect(`/login?error=Please verify your email before logging in.&email=${encodeURIComponent(data.email)}`)
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function signup(formData: FormData) {
  const supabase = await createClient()

  const email = formData.get('email') as string
  const password = formData.get('password') as string

  const { error } = await supabase.auth.signUp({
    email,
    password,
  })

  if (error) {
    return redirect('/login?error=' + encodeURIComponent(error.message))
  }

  // After signup, redirect to the OTP verification screen
  redirect(`/login?email=${encodeURIComponent(email)}&verify=true&type=signup`)
}

export async function verifyOtp(formData: FormData) {
  const supabase = await createClient()
  const email = formData.get('email') as string
  const token = formData.get('token') as string
  const type = (formData.get('type') as 'signup' | 'login') || 'signup'

  const { error } = await supabase.auth.verifyOtp({
    email,
    token,
    type: 'signup',
  })

  if (error) {
    return redirect(`/login?email=${encodeURIComponent(email)}&verify=true&type=${type}&error=` + encodeURIComponent(error.message))
  }

  revalidatePath('/', 'layout')
  redirect('/')
}

export async function resendVerification(formData: FormData) {
  const supabase = await createClient()
  const email = formData.get('email') as string

  const { error } = await supabase.auth.resend({
    type: 'signup',
    email: email,
  })

  if (error) {
    return redirect('/login?error=' + encodeURIComponent(error.message))
  }

  redirect(`/login?email=${encodeURIComponent(email)}&verify=true&type=signup&message=New code sent!`)
}
export async function logout() {
  const supabase = await createClient()
  await supabase.auth.signOut()
  revalidatePath('/', 'layout')
  redirect('/login')
}
