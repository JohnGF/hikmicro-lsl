# gateway.py
import asyncio
from fastapi import FastAPI, WebSocket
from pylsl import resolve_byprop, StreamInlet
import numpy as np

app = FastAPI()

@app.websocket("/ws/thermal")
async def thermal_stream(websocket: WebSocket):
    await websocket.accept()
    
    streams = resolve_byprop("name", "HIKMICRO", timeout=3.0)
    if not streams:
        await websocket.close(code=1000, reason="No stream")
        return
        
    inlet = StreamInlet(streams[0])
    
    loop = asyncio.get_running_loop()
    try:
        while True:
            # Non-blocking pull in executor
            sample, ts = await loop.run_in_executor(None, inlet.pull_sample, 0.05)
            if sample is not None:
                arr = np.array(sample, dtype=np.float32)
                # Send summary stats or send the binary buffer directly
                await websocket.send_json({
                    "fps": 25.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "center": float(arr[len(arr) // 2]),
                })
            await asyncio.sleep(0.01)
    except Exception:
        pass
