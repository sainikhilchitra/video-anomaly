"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export function useWebSocket(url: string) {
  const [isReady, setIsReady] = useState(false);
  const [val, setVal] = useState<any>(null);
  const [status, setStatus] = useState<string>("disconnected");
  
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(url);
    ws.current = socket;

    socket.onopen = () => {
      setIsReady(true);
      setStatus("connected");
      console.log("WebSocket connected");
    };

    socket.onclose = () => {
      setIsReady(false);
      setStatus("disconnected");
      console.log("WebSocket disconnected");
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setVal(data);
    };

    return () => {
      socket.close();
    };
  }, [url]);

  const send = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data));
    }
  }, []);

  return { isReady, val, send, status };
}
