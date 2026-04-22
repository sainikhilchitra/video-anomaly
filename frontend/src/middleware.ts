import { NextResponse, type NextRequest } from 'next/server'
import { updateSession } from '@/utils/supabase/middleware'

export async function middleware(request: NextRequest) {
  const { supabase, supabaseResponse } = await updateSession(request)
  
  const { data: { user } } = await supabase.auth.getUser()

  // 1. If no user, redirect to /login
  if (!user && !request.nextUrl.pathname.startsWith('/login')) {
    return NextResponse.redirect(new URL('/login', request.url))
  }

  // 2. ENFORCE EMAIL VERIFICATION: If user exists but hasn't confirmed email
  if (user && !user.email_confirmed_at && !request.nextUrl.pathname.startsWith('/login')) {
    // We sign them out to clear the unverified session
    await supabase.auth.signOut()
    return NextResponse.redirect(new URL('/login?error=Please verify your email before logging in.', request.url))
  }

  // 3. If logged in and verified, don't let them go to /login
  if (user && user.email_confirmed_at && request.nextUrl.pathname.startsWith('/login')) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return supabaseResponse
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * Feel free to modify this pattern to include more paths.
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
