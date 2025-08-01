"""Web server for Novel AI Agent interface"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..utils.config import Config

class WebServer:
    """Web interface for monitoring and controlling the Novel AI Agent"""
    
    def __init__(self, novel_agent, config: Config):
        self.novel_agent = novel_agent
        self.config = config
        self.app = FastAPI(title="Novel AI Agent", description="Self-improving novel generation system")
        self.templates = Jinja2Templates(directory="templates")
        self.active_connections = []
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        if self.config.web_interface.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self):
        """Setup web routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard"""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "Novel AI Agent Dashboard"
            })
        
        @self.app.get("/api/status")
        async def get_status():
            """Get current system status"""
            try:
                status = self.novel_agent.get_status()
                return {"success": True, "data": status}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/api/story/progress")
        async def get_story_progress():
            """Get story generation progress"""
            try:
                story_state = self.novel_agent.story_state
                return {
                    "success": True,
                    "data": {
                        "word_count": story_state.get("current_word_count", 0),
                        "target_length": self.config.story.target_length,
                        "current_chapter": story_state.get("current_chapter", 1),
                        "progress_percentage": (story_state.get("current_word_count", 0) / self.config.story.target_length) * 100,
                        "recent_content": story_state.get("story_content", [])[-3:] if story_state.get("story_content") else []
                    }
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/api/characters")
        async def get_characters():
            """Get character information"""
            try:
                characters_data = []
                for character in self.novel_agent.characters:
                    characters_data.append(character.get_character_summary())
                return {"success": True, "data": characters_data}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/api/world")
        async def get_world_info():
            """Get world simulation information"""
            try:
                if self.novel_agent.world_simulation:
                    world_data = self.novel_agent.world_simulation.get_world_summary()
                    return {"success": True, "data": world_data}
                else:
                    return {"success": False, "error": "World simulation not initialized"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/api/evolution")
        async def get_evolution_status():
            """Get code evolution status"""
            try:
                if self.novel_agent.code_evolver:
                    evolution_data = self.novel_agent.code_evolver.get_evolution_status()
                    return {"success": True, "data": evolution_data}
                else:
                    return {"success": False, "error": "Code evolution not enabled"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/api/control/start")
        async def start_generation():
            """Start novel generation"""
            try:
                # Start generation in background
                asyncio.create_task(self.novel_agent.generate_novel())
                return {"success": True, "message": "Novel generation started"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/api/control/evolve")
        async def trigger_evolution():
            """Manually trigger code evolution"""
            try:
                if self.novel_agent.code_evolver:
                    asyncio.create_task(self.novel_agent.code_evolver.evolve_system(self.novel_agent.story_state))
                    return {"success": True, "message": "Code evolution triggered"}
                else:
                    return {"success": False, "error": "Code evolution not enabled"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    status = self.novel_agent.get_status()
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status
                    })
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections.remove(conn)
    
    async def start(self):
        """Start the web server"""
        print(f"Starting web server on {self.config.web_interface.host}:{self.config.web_interface.port}")
        
        # Initialize the novel agent
        await self.novel_agent.initialize()
        
        # Start the server
        config = uvicorn.Config(
            self.app,
            host=self.config.web_interface.host,
            port=self.config.web_interface.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()