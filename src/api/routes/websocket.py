"""
WebSocket Routes for Real-time Communication

Real-time features:
- Live code collaboration
- Real-time AI assistance
- Code execution streaming
- Live error detection
"""

import json
import logging
from typing import Dict, Any, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from ...core.ai_engine import ai_engine
from ...core.config import settings


logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.rooms: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Remove from all rooms
        for room_clients in self.rooms.values():
            room_clients.discard(client_id)
        
        logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to {client_id}: {e}")
                    self.disconnect(client_id)
    
    async def broadcast_to_room(self, message: str, room_id: str, exclude: str = None):
        """Broadcast a message to all clients in a room."""
        if room_id in self.rooms:
            for client_id in self.rooms[room_id].copy():
                if client_id != exclude and client_id in self.active_connections:
                    await self.send_personal_message(message, client_id)
    
    def join_room(self, client_id: str, room_id: str):
        """Add a client to a room."""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(client_id)
        logger.info(f"Client {client_id} joined room {room_id}")
    
    def leave_room(self, client_id: str, room_id: str):
        """Remove a client from a room."""
        if room_id in self.rooms:
            self.rooms[room_id].discard(client_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
        logger.info(f"Client {client_id} left room {room_id}")


# Global connection manager
manager = ConnectionManager()


@router.websocket("/coding/{client_id}")
async def websocket_coding_session(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for coding sessions."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            await handle_websocket_message(websocket, client_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected from coding session")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)


async def handle_websocket_message(websocket: WebSocket, client_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    try:
        message_type = message.get("type")
        data = message.get("data", {})
        
        if message_type == "ping":
            # Health check
            await send_response(websocket, client_id, "pong", {"timestamp": data.get("timestamp")})
        
        elif message_type == "join_room":
            # Join collaboration room
            room_id = data.get("room_id")
            if room_id:
                manager.join_room(client_id, room_id)
                await send_response(websocket, client_id, "room_joined", {"room_id": room_id})
                await manager.broadcast_to_room(
                    json.dumps({
                        "type": "user_joined",
                        "data": {"user_id": client_id, "room_id": room_id}
                    }),
                    room_id,
                    exclude=client_id
                )
        
        elif message_type == "leave_room":
            # Leave collaboration room
            room_id = data.get("room_id")
            if room_id:
                manager.leave_room(client_id, room_id)
                await send_response(websocket, client_id, "room_left", {"room_id": room_id})
                await manager.broadcast_to_room(
                    json.dumps({
                        "type": "user_left",
                        "data": {"user_id": client_id, "room_id": room_id}
                    }),
                    room_id
                )
        
        elif message_type == "code_change":
            # Real-time code collaboration
            if settings.features.enable_real_time_collaboration:
                room_id = data.get("room_id")
                if room_id:
                    await manager.broadcast_to_room(
                        json.dumps({
                            "type": "code_update",
                            "data": {
                                "user_id": client_id,
                                "changes": data.get("changes"),
                                "timestamp": data.get("timestamp")
                            }
                        }),
                        room_id,
                        exclude=client_id
                    )
        
        elif message_type == "ai_complete":
            # Real-time AI code completion
            if settings.features.enable_code_completion:
                await handle_ai_completion(websocket, client_id, data)
        
        elif message_type == "ai_analyze":
            # Real-time AI code analysis
            if settings.features.enable_code_analysis:
                await handle_ai_analysis(websocket, client_id, data)
        
        elif message_type == "cursor_position":
            # Cursor position sharing
            room_id = data.get("room_id")
            if room_id and settings.features.enable_real_time_collaboration:
                await manager.broadcast_to_room(
                    json.dumps({
                        "type": "cursor_update",
                        "data": {
                            "user_id": client_id,
                            "position": data.get("position"),
                            "selection": data.get("selection")
                        }
                    }),
                    room_id,
                    exclude=client_id
                )
        
        else:
            await send_error(websocket, client_id, f"Unknown message type: {message_type}")
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await send_error(websocket, client_id, f"Message processing error: {str(e)}")


async def handle_ai_completion(websocket: WebSocket, client_id: str, data: Dict[str, Any]):
    """Handle real-time AI code completion."""
    try:
        content = data.get("content", "")
        language = data.get("language", "python")
        
        if not content:
            await send_error(websocket, client_id, "No content provided for completion")
            return
        
        # Send processing status
        await send_response(websocket, client_id, "ai_processing", {"task": "completion"})
        
        # Get AI completion
        response = await ai_engine.complete_code(content, language)
        
        if response.success:
            await send_response(websocket, client_id, "ai_completion_result", {
                "content": response.content,
                "provider": response.provider.value,
                "model": response.model
            })
        else:
            await send_error(websocket, client_id, f"AI completion failed: {response.error}")
    
    except Exception as e:
        logger.error(f"AI completion error: {e}")
        await send_error(websocket, client_id, f"AI completion error: {str(e)}")


async def handle_ai_analysis(websocket: WebSocket, client_id: str, data: Dict[str, Any]):
    """Handle real-time AI code analysis."""
    try:
        content = data.get("content", "")
        language = data.get("language", "python")
        
        if not content:
            await send_error(websocket, client_id, "No content provided for analysis")
            return
        
        # Send processing status
        await send_response(websocket, client_id, "ai_processing", {"task": "analysis"})
        
        # Get AI analysis
        response = await ai_engine.analyze_code(content, language)
        
        if response.success:
            await send_response(websocket, client_id, "ai_analysis_result", {
                "content": response.content,
                "provider": response.provider.value,
                "model": response.model
            })
        else:
            await send_error(websocket, client_id, f"AI analysis failed: {response.error}")
    
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        await send_error(websocket, client_id, f"AI analysis error: {str(e)}")


async def send_response(websocket: WebSocket, client_id: str, response_type: str, data: Dict[str, Any]):
    """Send a response message to the client."""
    try:
        message = json.dumps({
            "type": response_type,
            "data": data,
            "timestamp": int(time.time() * 1000)
        })
        await manager.send_personal_message(message, client_id)
    except Exception as e:
        logger.error(f"Error sending response to {client_id}: {e}")


async def send_error(websocket: WebSocket, client_id: str, error_message: str):
    """Send an error message to the client."""
    await send_response(websocket, client_id, "error", {"message": error_message})


# Additional WebSocket endpoints
@router.websocket("/collaboration/{room_id}/{client_id}")
async def websocket_collaboration(websocket: WebSocket, room_id: str, client_id: str):
    """WebSocket endpoint for real-time collaboration."""
    if not settings.features.enable_real_time_collaboration:
        await websocket.close(code=1008, reason="Real-time collaboration is disabled")
        return
    
    await manager.connect(websocket, client_id)
    manager.join_room(client_id, room_id)
    
    # Notify others in the room
    await manager.broadcast_to_room(
        json.dumps({
            "type": "user_joined",
            "data": {"user_id": client_id, "room_id": room_id}
        }),
        room_id,
        exclude=client_id
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Broadcast to room
            await manager.broadcast_to_room(
                json.dumps({
                    "type": "collaboration_update",
                    "data": {
                        "user_id": client_id,
                        "message": message,
                        "timestamp": int(time.time() * 1000)
                    }
                }),
                room_id,
                exclude=client_id
            )
            
    except WebSocketDisconnect:
        manager.leave_room(client_id, room_id)
        manager.disconnect(client_id)
        
        # Notify others in the room
        await manager.broadcast_to_room(
            json.dumps({
                "type": "user_left",
                "data": {"user_id": client_id, "room_id": room_id}
            }),
            room_id
        )
        
        logger.info(f"Client {client_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"Collaboration WebSocket error for client {client_id}: {e}")
        manager.leave_room(client_id, room_id)
        manager.disconnect(client_id)