import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import CognitiveGauge from './components/CognitiveGauge';
import ArousalValencePlot from './components/ArousalValencePlot';
import TrajectoryChart from './components/TrajectoryChart';
import AlertOverlay from './components/AlertOverlay';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8765';
const RECONNECT_DELAY_MS = 3000;

const INITIAL_STATE = {
  predicted_class: 1,
  class_name: 'Optimal',
  confidence: 0.5,
  class_probabilities: { Underload: 0.1, Optimal: 0.6, Overload: 0.2, Fatigue: 0.1 },
  arousal: 0.5,
  valence: 0.5,
  modality_contributions: { eeg: 0.5, eye: 0.3, hrv: 0.2 },
  per_modality_predictions: { eeg: 'Optimal', eye: 'Optimal', hrv: 'Optimal' },
  modality_agreement: 1.0,
};

export default function App() {
  const [cogState, setCogState] = useState(INITIAL_STATE);
  const [trajectory, setTrajectory] = useState([]);
  const [adaptationPolicy, setAdaptationPolicy] = useState({ mode: 'normal', actions: [], alert: null });
  const [connected, setConnected] = useState(false);
  const [alertVisible, setAlertVisible] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state_update') {
          setCogState(msg.state);
          setAdaptationPolicy(msg.adaptation_policy || { mode: 'normal', actions: [], alert: null });
          setTrajectory(prev => {
            const entry = {
              timestamp: msg.timestamp,
              predicted_class: msg.state.predicted_class,
              confidence: msg.state.confidence,
              arousal: msg.state.arousal,
              valence: msg.state.valence,
            };
            const next = [...prev, entry];
            return next.slice(-60); // keep last 30s at 2Hz
          });
          if (msg.adaptation_policy?.alert) {
            setAlertVisible(true);
          }
        } else if (msg.type === 'trajectory_init') {
          setTrajectory(msg.trajectory || []);
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  const layoutClass = adaptationPolicy.mode === 'simplified'
    ? 'layout simplified'
    : adaptationPolicy.mode === 'engagement'
    ? 'layout engagement'
    : 'layout';

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">
          <span className="title-icon">🧠</span> CognitiveTwin
        </h1>
        <div className="header-right">
          <div className={`connection-indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '● Live' : '○ Connecting…'}
          </div>
          <div className={`state-badge state-${cogState.class_name?.toLowerCase()}`}>
            {cogState.class_name} ({(cogState.confidence * 100).toFixed(0)}%)
          </div>
        </div>
      </header>

      <main className={layoutClass}>
        <section className="panel panel-gauge">
          <h2 className="panel-title">Cognitive Load</h2>
          <CognitiveGauge
            value={cogState.predicted_class}
            confidence={cogState.confidence}
            classProbs={cogState.class_probabilities}
          />
          <div className="modality-contributions">
            {Object.entries(cogState.modality_contributions || {}).map(([mod, w]) => (
              <div key={mod} className="modality-bar">
                <span className="modality-label">{mod.toUpperCase()}</span>
                <div className="modality-bar-fill" style={{ width: `${w * 100}%` }} />
                <span className="modality-value">{(w * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel panel-av">
          <h2 className="panel-title">Arousal–Valence</h2>
          <ArousalValencePlot
            arousal={cogState.arousal}
            valence={cogState.valence}
            history={trajectory.slice(-20)}
          />
        </section>

        <section className="panel panel-trajectory">
          <h2 className="panel-title">30s Trajectory</h2>
          <TrajectoryChart trajectory={trajectory} />
        </section>

        {adaptationPolicy.mode === 'engagement' && (
          <section className="panel panel-engagement">
            <h2 className="panel-title">Engagement Cues</h2>
            <p className="engagement-message">
              Cognitive load is low — consider engaging with additional tasks or
              increasing task complexity.
            </p>
          </section>
        )}
      </main>

      {alertVisible && adaptationPolicy.alert && (
        <AlertOverlay
          alert={adaptationPolicy.alert}
          onDismiss={() => setAlertVisible(false)}
        />
      )}
    </div>
  );
}
