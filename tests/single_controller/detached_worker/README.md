# Detached Worker
## How to run (Only on a single node)
- Start a local ray cluster: 
```bash
ray start --head --port=6379
```
- Run the server
```bash
python3 server.py
```
- On another terminal, Run the client
```bash
python3 client.py
```
