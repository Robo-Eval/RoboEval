import asyncio
import websockets
import json

CHANNEL = "roboeval_sim_channel"
UID = 12345

async def handler(websocket):
    print("Client connected!")

    join_msg = {
        "jsonrpc": "2.0",
        "method": "join",
        "params": {
            "channel": CHANNEL,
            "uid": UID
        },
        "id": 1
    }

    await websocket.send(json.dumps(join_msg))

    async for message in websocket:
        print("Received from client:", message)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Server running at ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())
