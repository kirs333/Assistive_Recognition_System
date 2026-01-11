import { useEffect, useRef, useState, useCallback } from 'react';

const useWebSocket = (url, options = {}) => {
    const [lastMessage, setLastMessage] = useState(null);
    const [readyState, setReadyState] = useState(WebSocket.CLOSED);

    const ws = useRef(null);
    const reconnectTimeout = useRef(null);
    const shouldReconnect = useRef(true);
    const isConnecting = useRef(false);
    const hasMountedOnce = useRef(false);

    const {
        onOpen,
        onClose,
        onError,
        reconnectInterval = 5000,
    } = options;

    const connect = useCallback(() => {
        if (isConnecting.current || ws.current) return;

        isConnecting.current = true;
        console.log('Connecting to:', url);

        try {
            const socket = new WebSocket(url);
            ws.current = socket;

            socket.onopen = (event) => {
                console.log('WebSocket connected');
                setReadyState(WebSocket.OPEN);
                isConnecting.current = false;
                onOpen?.(event);

                if (reconnectTimeout.current) {
                    clearTimeout(reconnectTimeout.current);
                    reconnectTimeout.current = null;
                }
            };

            socket.onmessage = (event) => {
                setLastMessage(event.data);
            };

            socket.onerror = (event) => {
                console.error('WebSocket error', event);
                onError?.(event);

            };

            socket.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                setReadyState(WebSocket.CLOSED);
                ws.current = null;
                isConnecting.current = false;
                onClose?.(event);

                if (shouldReconnect.current && event.code !== 1000) {
                    reconnectTimeout.current = setTimeout(connect, reconnectInterval);
                }
            };
        } catch (err) {
            console.error('WebSocket creation failed', err);
            isConnecting.current = false;
        }
    }, [url, onOpen, onClose, onError, reconnectInterval]);

    useEffect(() => {
        if (!hasMountedOnce.current) {
            hasMountedOnce.current = true;
            connect();
        }

        return () => {
            shouldReconnect.current = false;

            if (reconnectTimeout.current) {
                clearTimeout(reconnectTimeout.current);
            }

            // 🔒 Close only real connections
            if (ws.current && ws.current.readyState === WebSocket.OPEN) {
                ws.current.close(1000, 'Component unmounted');
            }

            ws.current = null;
        };
    }, [connect]);

    const sendMessage = useCallback((data) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket not open');
        }
    }, []);

    const sendCommand = useCallback((command) => {
        sendMessage({ type: 'command', command });
    }, [sendMessage]);

    return {
        sendMessage,
        sendCommand,
        lastMessage,
        readyState,
    };
};

export default useWebSocket;
