# Deployment Guide: Video Anomaly Detection

This guide explains how to deploy the **Backend** to Render and the **Frontend** to Vercel.

---

## 🚀 Pre-requisites
1. Push your code to a GitHub repository.
2. Ensure `.gitignore` is present (already added).

---

## 🛠 Backend: Render Deployment
1. Log in to [Render](https://render.com).
2. Click **New +** > **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically detect `render.yaml`. 
5. Click **Apply**.

### 🔧 Manual Configuration (If not using Blueprint)
- **Service Type**: Web Service
- **Runtime**: Python 3
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.10.12` (or similar)
  - `MODEL_PATH`: `../scripts/attention_Avenue.pth`

---

## 🌐 Frontend: Vercel Deployment
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New** > **Project**.
3. Import your GitHub repository.
4. **Project Settings**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
5. **Environment Variables**:
   - Add `NEXT_PUBLIC_WS_URL`: `wss://your-backend-name.onrender.com/ws`
   - *Note: Use `wss://` for production (HTTPS).*
6. Click **Deploy**.

---

## ⚙️ Local Build & Run Commands

### 🐍 Backend (Local)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
*Runs on `http://localhost:8000`*

### ⚛️ Frontend (Local)
```bash
cd frontend
npm install
npm run dev
```
*Runs on `http://localhost:3000`*

### 🏗️ Frontend (Production Build Test)
```bash
cd frontend
npm run build
npm run start
```

---

## ⚠️ Important Notes
- **WebSockets**: Ensure you are on a Render plan that supports persistent connections (Starter tier recommended) to avoid 502 errors during spin-down.
- **CORS**: The backend currently allows all origins (`*`). For strict security, update `allow_origins` in `backend/main.py` to your Vercel URL.
