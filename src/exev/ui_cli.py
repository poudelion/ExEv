"""Command-line launcher for the local ExEv digital-twin dashboard."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .disasters import DISASTER_PROFILES
from .experiments import ExperimentConfig
from .routing import ROUTER_NAMES
from .visualization import build_dashboard_payload


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ExEv digital-twin dashboard")
    parser.add_argument("--agents", type=_positive, default=500)
    parser.add_argument("--width", type=_positive, default=10)
    parser.add_argument("--height", type=_positive, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disaster", choices=DISASTER_PROFILES, default="fire")
    parser.add_argument(
        "--routers", nargs="+", choices=ROUTER_NAMES,
        default=["dijkstra", "hazard-aware"],
        help="one or two unique algorithms",
    )
    parser.add_argument("--frame-every", type=_positive, default=2)
    parser.add_argument("--render-agents", type=_positive, default=1_000)
    parser.add_argument("--max-ticks", type=_positive, default=10_000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the browser")
    parser.add_argument("--export", type=Path, help="write dashboard JSON and exit")
    return parser


def _serve(
    payload: dict,
    port: int,
    open_browser: bool,
    *,
    frame_interval: int,
    max_rendered_agents: int,
) -> None:
    assets = files("exev.web")
    config = ExperimentConfig(**payload["configuration"])
    initial_algorithms = tuple(run["algorithm"] for run in payload["runs"])
    initial_disaster = config.disaster_profile
    initial_key = (initial_disaster, initial_algorithms)
    cache = {initial_key: json.dumps(payload, separators=(",", ":")).encode()}

    def simulation_data(disaster: str, algorithms: tuple[str, ...]) -> bytes:
        key = (disaster, algorithms)
        if key not in cache:
            generated = build_dashboard_payload(
                replace(config, disaster_profile=disaster),
                list(algorithms),
                frame_interval=frame_interval,
                max_rendered_agents=max_rendered_agents,
            )
            cache[key] = json.dumps(generated, separators=(",", ":")).encode()
        return cache[key]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            request = urlsplit(self.path)
            route = request.path
            if route == "/api/simulation":
                query = parse_qs(request.query)
                requested = query.get("routers", [",".join(initial_algorithms)])[0]
                algorithms = tuple(name for name in requested.split(",") if name)
                disaster = query.get("disaster", [initial_disaster])[0]
                if (
                    not 1 <= len(algorithms) <= 2
                    or len(set(algorithms)) != len(algorithms)
                    or any(name not in ROUTER_NAMES for name in algorithms)
                    or disaster not in DISASTER_PROFILES
                ):
                    body = json.dumps({"error": "choose valid hazard and router options"}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body, content_type = simulation_data(disaster, algorithms), "application/json"
            elif route == "/api/config":
                body = json.dumps({"routers": ROUTER_NAMES, "disasters": DISASTER_PROFILES}).encode()
                content_type = "application/json"
            else:
                names = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
                name = names.get(route)
                if name is None:
                    self.send_error(404)
                    return
                body = assets.joinpath(name).read_bytes()
                content_type = "text/html" if name.endswith("html") else (
                    "text/css" if name.endswith("css") else "text/javascript"
                )
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"ExEv dashboard ready at {url}", file=sys.stderr, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.width < 2 or args.height < 2:
        parser.error("grid dimensions must be at least 2")
    if len(args.routers) not in (1, 2) or len(set(args.routers)) != len(args.routers):
        parser.error("--routers requires one or two unique algorithms")
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535")
    config = ExperimentConfig(
        width=args.width,
        height=args.height,
        agent_count=args.agents,
        seed=args.seed,
        max_ticks=args.max_ticks,
        disaster_profile=args.disaster,
    )
    try:
        payload = build_dashboard_payload(
            config,
            args.routers,
            frame_interval=args.frame_every,
            max_rendered_agents=args.render_agents,
        )
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(args.export.resolve())
            return
        _serve(
            payload,
            args.port,
            args.open,
            frame_interval=args.frame_every,
            max_rendered_agents=args.render_agents,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"exev-ui: {exc}\n")


if __name__ == "__main__":
    main()
