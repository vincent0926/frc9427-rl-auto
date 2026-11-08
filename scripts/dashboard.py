"""訓練監控網頁的本機伺服器：python scripts/dashboard.py，然後打開 http://localhost:8765

只讀 runs/ 底下的檔案，不控制訓練。網頁每幾秒重新抓一次資料。
"""

import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard/index.html"
        if self.path in ("/api/runs", "/dashboard/api/runs"):
            runs = []
            if RUNS.exists():
                for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    status = d / "status.json"
                    if status.exists():
                        try:
                            runs.append(json.loads(status.read_text(encoding="utf-8")))
                        except json.JSONDecodeError:
                            pass
            body = json.dumps(runs).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not (self.path.startswith("/dashboard/") or self.path.startswith("/runs/")):
            self.send_error(404)
            return
        super().do_GET()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(ROOT)))
    print(f"訓練監控：http://localhost:{PORT}  (Ctrl+C 結束)")
    server.serve_forever()
