#!/usr/bin/env python3
"""
Central Bot Manager - Spawns 100 Individual Auction Bots

This script creates 100 separate bot processes, each with:
- A target auction rank (1-10, where 1 = best auction)
- A bid timing cycle (1-10, determines which round of every 10 they bid)

Configuration:
- 10 bots for each auction rank (1st best through 10th best)
- Each bot bids once every 10 rounds on their assigned turn
- Bots are distributed across turns 1-10 to ensure continuous bidding
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import List, Dict

class BotManager:
    def __init__(self):
        self.bot_processes: List[subprocess.Popen] = []
        self.base_dir = Path(__file__).parent.absolute()
        self.individual_bot_script = self.base_dir / "individual_bot.py"
        self.running = False
        
    def generate_bot_configs(self) -> List[Dict[str, int]]:
        """Generate configurations for 100 bots"""
        configs = []
        
        # Create 100 bots: 10 for each auction rank (0-9)
        for auction_rank in range(10):  # 0-9 (0=best auction, 9=10th best)
            for turn_in_cycle in range(1, 11):  # 1-10 (which turn of the 10-round cycle)
                bot_id = auction_rank * 10 + (turn_in_cycle - 1)  # 0-99
                
                config = {
                    'bot_id': bot_id,
                    'target_auction_rank': auction_rank,
                    'bid_turn_in_cycle': turn_in_cycle
                }
                configs.append(config)
        
        return configs
    
    def start_individual_bot(self, config: Dict[str, int], delay: float = 0) -> subprocess.Popen:
        """Start a single bot process"""
        bot_id = config['bot_id']
        target_rank = config['target_auction_rank']
        bid_turn = config['bid_turn_in_cycle']
        
        if delay > 0:
            time.sleep(delay)
        
        cmd = [
            sys.executable,
            str(self.individual_bot_script),
            str(bot_id),
            str(target_rank),
            str(bid_turn)
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            
            print(f"✅ Started Bot #{bot_id:2d} (Rank {target_rank+1:2d}, Turn {bid_turn:2d})")
            return process
            
        except Exception as e:
            print(f"❌ Failed to start Bot #{bot_id}: {e}")
            return None
    
    def start_all_bots(self, stagger_delay: float = 0.2) -> bool:
        """Start all 100 bots with optional staggered delays"""
        print("🚀 Starting 100 Individual Auction Bots...")
        print("=" * 60)
        
        # Check if individual bot script exists
        if not self.individual_bot_script.exists():
            print(f"❌ Individual bot script not found: {self.individual_bot_script}")
            return False
        
        # Generate bot configurations
        bot_configs = self.generate_bot_configs()
        print(f"📋 Generated {len(bot_configs)} bot configurations")
        print()
        
        # Display configuration summary
        print("🎯 Bot Distribution:")
        for rank in range(10):
            bots_for_rank = [c for c in bot_configs if c['target_auction_rank'] == rank]
            turns = [c['bid_turn_in_cycle'] for c in bots_for_rank]
            print(f"  Auction Rank {rank+1:2d}: {len(bots_for_rank)} bots (turns {min(turns)}-{max(turns)})")
        print()
        
        # Start all bots
        successful_starts = 0
        for i, config in enumerate(bot_configs):
            process = self.start_individual_bot(config, stagger_delay)
            if process:
                self.bot_processes.append(process)
                successful_starts += 1
            
            # Show progress every 20 bots
            if (i + 1) % 20 == 0:
                print(f"📊 Progress: {i+1}/{len(bot_configs)} bots started")
        
        print()
        print(f"✅ Successfully started {successful_starts}/{len(bot_configs)} bots")
        
        if successful_starts > 0:
            print("⏳ Waiting for bots to connect to server...")
            time.sleep(3)
            self.running = True
            return True
        else:
            print("❌ No bots started successfully")
            return False
    
    def monitor_bots(self):
        """Monitor bot processes and show status"""
        print("\n📊 Bot Status Monitor")
        print("=" * 40)
        
        while self.running:
            try:
                alive_count = 0
                dead_count = 0
                
                for process in self.bot_processes:
                    if process.poll() is None:
                        alive_count += 1
                    else:
                        dead_count += 1
                
                print(f"\r🤖 Alive: {alive_count:3d} | 💀 Dead: {dead_count:3d} | Total: {len(self.bot_processes):3d}", end="", flush=True)
                
                time.sleep(5)  # Update every 5 seconds
                
            except KeyboardInterrupt:
                print("\n🛑 Monitor interrupted")
                break
    
    def cleanup_bots(self):
        """Clean up all bot processes"""
        print("\n🧹 Cleaning up bot processes...")
        
        self.running = False
        terminated_count = 0
        killed_count = 0
        
        for i, process in enumerate(self.bot_processes):
            if process.poll() is None:  # Process is still running
                try:
                    # Try graceful termination first
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                        terminated_count += 1
                    except subprocess.TimeoutExpired:
                        # Force kill if termination doesn't work
                        process.kill()
                        process.wait()
                        killed_count += 1
                except Exception as e:
                    print(f"❌ Error stopping bot {i}: {e}")
        
        print(f"✅ Cleanup complete: {terminated_count} terminated, {killed_count} killed")
    
    def run_bot_army(self, monitor: bool = True):
        """Main method to run the entire bot army"""
        try:
            print("🎲 Central Bot Manager - 100 Bot Army")
            print("=" * 50)
            print()
            
            # Start all bots
            if not self.start_all_bots():
                return False
            
            print("🎉 All bots started successfully!")
            print("💡 Bot army is now running and bidding on auctions")
            print()
            
            if monitor:
                print("📊 Starting status monitor...")
                print("💡 Press Ctrl+C to stop all bots")
                self.monitor_bots()
            else:
                print("💡 Press Ctrl+C to stop all bots")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping bot army...")
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Bot army interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
        finally:
            self.cleanup_bots()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Central Bot Manager - 100 Bot Army")
    parser.add_argument("--no-monitor", action="store_true",
                       help="Don't show real-time bot status monitor")
    parser.add_argument("--delay", type=float, default=0.1,
                       help="Delay between starting bots (seconds, default: 0.1)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show bot configuration without starting them")
    
    args = parser.parse_args()
    
    manager = BotManager()
    
    if args.dry_run:
        print("🔍 Dry Run - Bot Configuration Preview")
        print("=" * 50)
        
        configs = manager.generate_bot_configs()
        print(f"Total bots: {len(configs)}")
        print()
        
        # Show first few and last few configurations
        print("First 10 bot configurations:")
        for config in configs[:10]:
            print(f"  Bot {config['bot_id']:2d}: Rank {config['target_auction_rank']+1:2d}, Turn {config['bid_turn_in_cycle']:2d}")
        
        print("...")
        
        print("Last 10 bot configurations:")
        for config in configs[-10:]:
            print(f"  Bot {config['bot_id']:2d}: Rank {config['target_auction_rank']+1:2d}, Turn {config['bid_turn_in_cycle']:2d}")
        
        print()
        print("Distribution by auction rank:")
        for rank in range(10):
            bots_for_rank = [c for c in configs if c['target_auction_rank'] == rank]
            turns = sorted([c['bid_turn_in_cycle'] for c in bots_for_rank])
            print(f"  Rank {rank+1:2d}: {len(bots_for_rank)} bots (turns {turns})")
        
        return
    
    # Run the bot army
    success = manager.run_bot_army(monitor=not args.no_monitor)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()