"""Local no-cache HTTP server for previewing site/ without stale browser-cached HTML/JS/CSS."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve site/ locally with Cache-Control: no-store.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", type=str, default="127.0.0.1")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "site",
    )
    args = parser.parse_args()
    handler = partial(NoCacheRequestHandler, directory=str(args.directory))
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {args.directory} at http://{args.bind}:{args.port} (no-cache)...", flush=True)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
