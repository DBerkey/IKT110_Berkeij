#!/usr/bin/env python3
"""
DND Auction Game Launcher

This script automatically starts the auction server and connects your specified agents.
Configure your agent paths in the AGENT_CONFIGS list below.
"""

import os
import sys
import time
import subprocess
import asyncio
import signal
import threading
from pathlib import Path
from typing import List, Dict, Optional

# ==========================================
# CONFIGURATION SECTION
# ==========================================

# Configure your agents here
AGENT_CONFIGS = [
    {
        "name": "Enhanced Partial Split",
        "path": "dnd_auction_agents/enhanced_agent_partial_split.py",
        "enabled": True
    },
    {
        "name": "ML Predictor", 
        "path": "dnd_auction_agents/agent_ml_predictor",
        "enabled": True
    },
    {
        "name": "Smart Predictor",
        "path": "dnd_auction_agents/agent_smart_predictor", 
        "enabled": True
    },
    {
        "name": "Predictive Bidding",
        "path": "dnd_auction_agents/agent_predictive_bidding",
        "enabled": True
    },
    {
        "name": "Partial Split",
        "path": "dnd_auction_agents/agent_partial_split",
        "enabled": True
    },
    {
        "name": "Random Single",
        "path": "dnd_auction_game/example_agents/agent_random_single.py",
        "enabled": True
    },
    {
        "name": "Random Walk", 
        "path": "dnd_auction_game/example_agents/agent_random_walk.py",
        "enabled": True
    },
    {
        "name": "Tiny Bid",
        "path": "dnd_auction_game/example_agents/agent_tiny_bid.py", 
        "enabled": True
    }
]

# Game settings
DEFAULT_ROUNDS = 1000
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_TIME_PER_ROUND = 1.0

# Server settings
GAME_TOKEN = "play123"
PLAY_TOKEN = "play123"

class GameLauncher:
    def __init__(self):
        self.server_process = None
        self.agent_processes = []
        self.game_running = False
        self.base_dir = Path(__file__).parent.absolute()
        
    def start_server(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Start the DND auction server"""
        print(f"🚀 Starting DND Auction Server on {host}:{port}...")
        
        # Set environment variables
        env = os.environ.copy()
        env["AH_GAME_TOKEN"] = GAME_TOKEN
        env["AH_PLAY_TOKEN"] = PLAY_TOKEN
        
        # Start the server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "dnd_auction_game.server:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "warning"  # Reduce server output
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            cwd=self.base_dir / "dnd_auction_game",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'  # Replace unicode errors instead of failing
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        if self.server_process.poll() is None:
            print("✅ Server started successfully!")
            return True
        else:
            print("❌ Failed to start server")
            stdout, stderr = self.server_process.communicate()
            print(f"Error: {stderr}")  # No need to decode, already text
            return False
    
    def get_enabled_agents(self) -> List[Dict]:
        """Get list of enabled agents with validated paths"""
        enabled_agents = []
        
        for agent_config in AGENT_CONFIGS:
            if not agent_config.get("enabled", False):
                continue
                
            agent_path = self.base_dir / agent_config["path"]
            
            # Check if path exists
            if not agent_path.exists():
                print(f"⚠️  Warning: Agent path not found: {agent_path}")
                continue
                
            enabled_agents.append({
                "name": agent_config["name"],
                "path": str(agent_path),
                "is_file": agent_path.is_file()
            })
        
        return enabled_agents
    
    def start_agent(self, agent_config: Dict, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Start a single agent"""
        agent_name = agent_config["name"]
        agent_path = agent_config["path"]
        is_file = agent_config["is_file"]
        
        print(f"🤖 Starting agent: {agent_name}")
        
        if is_file:
            # If it's a Python file, run it directly
            cmd = [sys.executable, agent_path]
        else:
            # If it's a directory, look for main.py or __main__.py
            main_file = None
            agent_dir = Path(agent_path)
            
            for possible_main in ["main.py", "__main__.py", f"{agent_dir.name}.py"]:
                potential_path = agent_dir / possible_main
                if potential_path.exists():
                    main_file = str(potential_path)
                    break
            
            if main_file:
                cmd = [sys.executable, main_file]
            else:
                print(f"❌ Could not find main file for agent: {agent_name}")
                return None
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'  # Replace unicode errors instead of failing
            )
            
            # Give agent a moment to start
            time.sleep(0.5)
            
            if process.poll() is None:
                print(f"✅ Agent '{agent_name}' started successfully")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Agent '{agent_name}' failed to start")
                print(f"Error: {stderr}")  # No need to decode, already text
                return None
                
        except Exception as e:
            print(f"❌ Failed to start agent '{agent_name}': {e}")
            return None
    
    def start_all_agents(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Start all enabled agents"""
        enabled_agents = self.get_enabled_agents()
        
        if not enabled_agents:
            print("❌ No enabled agents found!")
            return False
        
        print(f"🎯 Starting {len(enabled_agents)} agents...")
        
        for agent_config in enabled_agents:
            process = self.start_agent(agent_config, host, port)
            if process:
                self.agent_processes.append(process)
        
        if self.agent_processes:
            print(f"✅ Started {len(self.agent_processes)} agents successfully")
            # Give all agents time to connect
            print("⏳ Waiting for agents to connect...")
            time.sleep(2)
            return True
        else:
            print("❌ No agents started successfully")
            return False
    
    def run_game(self, rounds: int = DEFAULT_ROUNDS, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """Run the auction game"""
        print(f"🎮 Starting game with {rounds} rounds...")
        
        cmd = [
            sys.executable, "-m", "dnd_auction_game.play", 
            str(rounds), PLAY_TOKEN
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir / "dnd_auction_game",
                timeout=rounds * 10 + 30  # Allow time for all rounds plus buffer
            )
            
            if result.returncode == 0:
                print("🎉 Game completed successfully!")
                return True
            else:
                print("❌ Game ended with errors")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Game timed out")
            return False
        except Exception as e:
            print(f"❌ Error running game: {e}")
            return False
    
    def cleanup(self):
        """Clean up all processes"""
        print("🧹 Cleaning up...")
        
        # Stop all agent processes
        for process in self.agent_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        # Stop server process
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        print("✅ Cleanup completed")
    
    def launch_game(self, rounds: int = DEFAULT_ROUNDS, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, keep_server_running: bool = False):
        """Main method to launch the complete game setup"""
        try:
            print("🎲 DND Auction Game Launcher")
            print("=" * 50)
            
            # Start server
            if not self.start_server(host, port):
                return False
            
            # Start agents
            if not self.start_all_agents(host, port):
                return False
            
            # Run the game
            if not self.run_game(rounds, host, port):
                return False
            
            time.sleep(rounds * 1.5)  # Give some time for final processing

            print("\n🏆 Game session completed!")
            print(f"📊 Check the leaderboard at: http://{host}:{port}/")
            print("📝 Check the logs in the ./logs directory")
            
            if keep_server_running:
                print("\n🖥️  Server is still running - you can view the leaderboard!")
                print(f"🌐 Leaderboard URL: http://{host}:{port}/")
                print("💡 Press Ctrl+C to stop the server when you're done")
                try:
                    # Keep the main thread alive while server runs
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping server...")
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Game interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
        finally:
            if not keep_server_running:
                self.cleanup()
            else:
                # Only clean up agents, keep server running
                print("🧹 Cleaning up agents...")
                for process in self.agent_processes:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                print("✅ Agent cleanup completed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DND Auction Game Launcher")
    parser.add_argument("--rounds", "-r", type=int, default=DEFAULT_ROUNDS,
                       help=f"Number of game rounds (default: {DEFAULT_ROUNDS})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                       help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT,
                       help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument("--keep-server", "-k", action="store_true",
                       help="Keep server running after game ends to view leaderboard")
    parser.add_argument("--server-only", "-s", action="store_true",
                       help="Start only the server (no agents, no game)")
    parser.add_argument("--list-agents", action="store_true",
                       help="List all configured agents and their status")
    
    args = parser.parse_args()
    
    launcher = GameLauncher()
    
    if args.list_agents:
        print("🤖 Configured Agents:")
        print("=" * 30)
        enabled_agents = launcher.get_enabled_agents()
        
        for i, config in enumerate(AGENT_CONFIGS, 1):
            status = "✅ ENABLED" if config.get("enabled", False) else "❌ DISABLED"
            path_exists = "📁" if (launcher.base_dir / config["path"]).exists() else "❌"
            print(f"{i:2d}. {config['name']:<20} {status} {path_exists} {config['path']}")
        
        print(f"\n📊 Total: {len(AGENT_CONFIGS)} configured, {len(enabled_agents)} enabled and valid")
        return
    
    # Handle different modes
    if args.server_only:
        # Start only the server
        print("🖥️  Starting server-only mode...")
        if launcher.start_server(args.host, args.port):
            print(f"🌐 Server running at: http://{args.host}:{args.port}/")
            print("💡 Press Ctrl+C to stop the server")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                launcher.cleanup()
        success = True
    else:
        # Run the full game
        success = launcher.launch_game(args.rounds, args.host, args.port, args.keep_server)
        if args.keep_server:
            launcher.cleanup()  # Final cleanup when user stops
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()