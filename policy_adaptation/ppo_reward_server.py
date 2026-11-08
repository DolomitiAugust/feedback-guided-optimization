"""Local HTTP reward API matching OpenRLHF v0.6.0 remote_rm_fn."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from ppo_reward import response_only, score_response


class RewardHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        if self.path != "/reward":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size))
            queries, prompts, labels = (payload[k] for k in ("query", "prompts", "labels"))
            if not all(isinstance(x, list) for x in (queries, prompts, labels)):
                raise ValueError("Batch fields must be lists")
            if not len(queries) == len(prompts) == len(labels):
                raise ValueError("Batch fields must have equal lengths")
            if not all(isinstance(x, str) for x in queries + prompts):
                raise ValueError("Queries and prompts must be strings")
            rewards = [score_response(response_only(q, p), str(y))
                       for q, p, y in zip(queries, prompts, labels)]
            data = json.dumps({"rewards": rewards}).encode("utf-8")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LocalRewardServer:
    """A task-owned localhost server; use an available port and always clean up."""
    def __enter__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RewardHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/reward"
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
