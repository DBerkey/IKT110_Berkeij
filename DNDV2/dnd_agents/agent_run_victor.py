import sys
from pathlib import Path

from dnd_auction_game.client import AuctionGameClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dnd_agents.agent_victor import make_bid
except ImportError as exc:
    raise ImportError(
        "Unable to import 'dnd_agents.agent_victor'. Ensure the repository root is on PYTHONPATH."
    ) from exc

def main():
    # Configuración por defecto
    host = "localhost"
    port = 8000
    token = "play123"
    agent_name = "AggressiveWarrior"
    player_id = "my_player_id"
    
    # Leer argumentos de línea de comandos
    if len(sys.argv) >= 2:
        host = sys.argv[1]
    if len(sys.argv) >= 3:
        player_id = sys.argv[2]
    if len(sys.argv) >= 4:
        agent_name = sys.argv[3]
    
    print(f"🤖 Starting bot: {agent_name}")
    print(f"🌐 Connecting to: {host}:{port}")
    print(f"🎮 Player ID: {player_id}")
    print("-" * 50)
    
    client = AuctionGameClient(
        host=host,
        port=port,
        agent_name=agent_name,
        token=token,
        player_id=player_id
    )
    
    # Ejecutar bot
    try:
        print("⚡ Bot started - Waiting for game...")
        client.run(make_bid)  # pasa la función correcta
        print("\n✅ Game finished!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
