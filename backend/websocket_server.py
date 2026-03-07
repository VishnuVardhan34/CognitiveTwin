"""WebSocket server broadcasting real-time cognitive state to frontend clients."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptation policy
# ---------------------------------------------------------------------------

def get_adaptation_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a UI adaptation policy from the current cognitive state.

    Args:
        state: Cognitive state dictionary (from ``declare_state``).

    Returns:
        Policy dictionary with ``mode``, ``actions``, and ``alert``.
    """
    class_name: str = state.get("class_name", "Optimal")
    confidence: float = state.get("confidence", 0.5)

    policy: Dict[str, Any] = {
        "mode": "normal",
        "actions": [],
        "alert": None,
    }

    if class_name == "Overload":
        policy["mode"] = "simplified"
        policy["actions"] = [
            "collapse_secondary_panels",
            "enlarge_critical_elements",
            "reduce_information_density",
        ]
        if confidence > 0.7:
            policy["alert"] = {
                "type": "overload",
                "severity": "high",
                "message": "Cognitive overload detected. Simplifying interface.",
                "modality": "visual",
            }

    elif class_name == "Underload":
        policy["mode"] = "engagement"
        policy["actions"] = [
            "introduce_engagement_cues",
            "increase_information_density",
            "enable_advanced_features",
        ]

    elif class_name == "Fatigue":
        policy["mode"] = "alert"
        policy["actions"] = [
            "trigger_break_reminder",
            "reduce_visual_complexity",
        ]
        policy["alert"] = {
            "type": "fatigue",
            "severity": "medium",
            "message": "Fatigue detected. Consider taking a break.",
            "modality": "auditory_visual",
        }

    return policy


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

class DigitalTwinServer:
    """WebSocket server that broadcasts cognitive state at 2 Hz.

    Maintains a 30-second trajectory buffer and sends combined state +
    adaptation policy to all connected clients.

    Args:
        host: Bind host. Default ``'localhost'``.
        port: Bind port. Default ``8765``.
        broadcast_rate_hz: Broadcast frequency. Default 2.0.
        trajectory_buffer_sec: Trajectory buffer duration in seconds. Default 30.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        broadcast_rate_hz: float = 2.0,
        trajectory_buffer_sec: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.broadcast_interval = 1.0 / broadcast_rate_hz
        self.trajectory_buffer_sec = trajectory_buffer_sec

        self._clients: Set[WebSocketServerProtocol] = set()
        self._current_state: Optional[Dict[str, Any]] = None
        self._trajectory: deque = deque(
            maxlen=int(broadcast_rate_hz * trajectory_buffer_sec)
        )

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    async def _register(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        logger.info("Client connected: %s (total: %d)", ws.remote_address, len(self._clients))
        # Send the current trajectory on connect
        if self._trajectory:
            await ws.send(
                json.dumps({
                    "type": "trajectory_init",
                    "trajectory": list(self._trajectory),
                })
            )

    async def _unregister(self, ws: WebSocketServerProtocol) -> None:
        self._clients.discard(ws)
        logger.info("Client disconnected (total: %d)", len(self._clients))

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def broadcast_state(self, state: Dict[str, Any]) -> None:
        """Queue a new cognitive state for broadcast.

        Args:
            state: Cognitive state dictionary from ``declare_state``.
        """
        policy = get_adaptation_policy(state)
        payload = {
            "type": "state_update",
            "timestamp": time.time(),
            "state": state,
            "adaptation_policy": policy,
        }
        self._current_state = payload
        self._trajectory.append({
            "timestamp": payload["timestamp"],
            "predicted_class": state.get("predicted_class", 0),
            "confidence": state.get("confidence", 0.0),
            "arousal": state.get("arousal", 0.5),
            "valence": state.get("valence", 0.5),
        })

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        await self._register(ws)
        try:
            async for message in ws:
                # Handle client messages (e.g., acknowledgements) if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._unregister(ws)

    # ------------------------------------------------------------------
    # Broadcast loop
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        while True:
            if self._current_state and self._clients:
                payload = json.dumps(self._current_state)
                dead = set()
                for ws in self._clients:
                    try:
                        await ws.send(payload)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(ws)
                self._clients -= dead
            await asyncio.sleep(self.broadcast_interval)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def serve(self) -> None:
        """Start the WebSocket server and broadcast loop."""
        logger.info("Starting CognitiveTwin WebSocket server on %s:%d", self.host, self.port)
        async with websockets.serve(self._handler, self.host, self.port):
            await self._broadcast_loop()

    def run(self) -> None:
        """Run the server (blocking)."""
        asyncio.run(self.serve())


def main() -> None:
    """CLI entry point for the WebSocket server."""
    logging.basicConfig(level=logging.INFO)
    server = DigitalTwinServer()
    server.run()


if __name__ == "__main__":
    main()
