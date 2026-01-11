import { useState, useEffect, useRef } from "react";
import "./App.css";
import useWebSocket from "./hooks/useWebSocket";
import VideoDisplay from "./components/VideoDisplay";
import CommandPanel from "./components/CommandPanel";
import StatusBar from "./components/StatusBar";
import VoiceCommands from "./components/VoiceCommands";

function App() {
  const [frameData, setFrameData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  const { sendCommand, lastMessage } = useWebSocket("ws://localhost:8000/ws", {
    onOpen: () => {
      console.log("WebSocket connected");
      setIsConnected(true);
    },
    onClose: () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    },
    onError: (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    },
  });

  // Handleing incoming messages
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const data = JSON.parse(lastMessage);
      console.log("Incoming frameData:", data);

      if (data.type === "frame") {
        setFrameData(data);
      } else if (data.type === "tts" && ttsEnabled) {
        // Using browser's Speech Synthesis API
        const utterance = new SpeechSynthesisUtterance(data.text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
      } else if (data.type === "ocr_result") {
        console.log("OCR Result:", data.text);
      }
    } catch (error) {
      console.error("Error parsing message:", error);
    }
  }, [lastMessage, ttsEnabled]);

  const handleCommand = (command) => {
    console.log("Sending command:", command);
    sendCommand(command);
  };

  const handleVoiceCommand = (command) => {
    console.log("Voice command detected:", command);
    handleCommand(command);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏥 Medicine Detection System</h1>
        <div className="connection-status">
          <span
            className={`status-indicator ${
              isConnected ? "connected" : "disconnected"
            }`}
          ></span>
          <span>{isConnected ? "Connected" : "Disconnected"}</span>
        </div>
      </header>

      <div className="app-content">
        <div className="video-section">
          <VideoDisplay frameData={frameData} />
        </div>

        <div className="control-section">
          <StatusBar
            mode={frameData?.mode || "SCAN"}
            fps={frameData?.fps || 0}
            objectCount={frameData?.detections?.length || 0}
            activeObject={frameData?.active_object}
          />

          <CommandPanel
            onCommand={handleCommand}
            isConnected={isConnected}
            currentMode={frameData?.mode || "SCAN"}
          />

          <div className="settings-panel">
            <h3>Settings</h3>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={ttsEnabled}
                onChange={(e) => setTtsEnabled(e.target.checked)}
              />
              <span>Text-to-Speech</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={voiceEnabled}
                onChange={(e) => setVoiceEnabled(e.target.checked)}
              />
              <span>Voice Commands</span>
            </label>
          </div>

          {voiceEnabled && (
            <VoiceCommands
              onCommand={handleVoiceCommand}
              isEnabled={voiceEnabled}
            />
          )}
        </div>
      </div>

      <footer className="app-footer">
        <p>
          Press buttons or use voice commands: "scan", "guide", "select", "read"
        </p>
      </footer>
    </div>
  );
}

export default App;
