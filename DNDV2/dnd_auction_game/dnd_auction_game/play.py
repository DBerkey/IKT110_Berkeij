import sys
import random
import asyncio
import json
import websockets


class AuctionGameRunner:
    def __init__(self, host:str, play_token:str, n_rounds=5, time_per_round:float=1.0, port:int=8000):
        self.host = host
        self.port = port
        self.n_rounds = n_rounds
        self.play_token = play_token
        
        self.time_per_round = time_per_round
        
    def run(self):
        asyncio.run(self._internal_run())
        
        
    async def _internal_run(self):
        connection_str = "ws://{}:{}/ws_run/{}".format(self.host, self.port, self.play_token)
        print("connecting to: {}".format(connection_str))
        

        async with websockets.connect(connection_str) as sock:
            print("<connected - starting game>")

            game_info = {"num_rounds": self.n_rounds}
            await sock.send(json.dumps(game_info))

            server_info_raw = await sock.recv()
            server_info = json.loads(server_info_raw)
            print("<server info: {}>".format(server_info))
            print("<game started>")

    
        print("<done>")
            
        
def main() -> None:
    host = "localhost"
    port = 8000

    n_rounds = 12
    if len(sys.argv) >= 2:
        n_rounds = int(sys.argv[1])

    if len(sys.argv) >= 3:
        play_token = sys.argv[2]
    else:
        play_token = "play123"

    if len(sys.argv) >= 4:
        host = sys.argv[3]

    if len(sys.argv) >= 5:
        port = int(sys.argv[4])

    runner = AuctionGameRunner(host, play_token=play_token, n_rounds=n_rounds, port=port)
    print("Running the game for: {} rounds on {}:{}.".format(n_rounds, host, port))
    runner.run()
    
    print("<game is done>")


if __name__ == "__main__":
    main()





