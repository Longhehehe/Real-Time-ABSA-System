"""Serve exactly one blinded assignment to the local annotation UI.

Run from the repository root:

    .\.venv\Scripts\python -X utf8 -m human_annotation_ui.serve \
      --assignment data\annotations\human_reference_v1_20260726\
      assignments\annotator_a.assignment.json --open
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import webbrowser

from human_annotation_ui.common import (
    IntegrityError,
    canonical_json,
    load_json,
    sha256_text,
    verify_assignment,
)
from human_annotation_ui.ai_review import verify_suggestions
from human_annotation_ui.adjudication import verify_adjudication


ROOT = Path(__file__).resolve().parent
STATIC_FILES = {
    "/": (ROOT / "index.html", "text/html; charset=utf-8"),
    "/index.html": (ROOT / "index.html", "text/html; charset=utf-8"),
    "/app.js": (ROOT / "app.js", "text/javascript; charset=utf-8"),
    "/annotation_core.mjs": (
        ROOT / "annotation_core.mjs",
        "text/javascript; charset=utf-8",
    ),
    "/styles.css": (ROOT / "styles.css", "text/css; charset=utf-8"),
}
MAX_STATIC_BYTES = 5 * 1024 * 1024


class AnnotationServer(ThreadingHTTPServer):
    """HTTP server with immutable, prevalidated assignment bytes."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        assignment_path: Path,
        guideline_path: Path,
        workflow_mode: str,
        suggestions_path: Path | None = None,
        adjudication_path: Path | None = None,
    ) -> None:
        assignment = verify_assignment(load_json(assignment_path))
        assignment_bytes = (
            json.dumps(
                assignment,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        guideline_bytes = guideline_path.read_bytes()
        if len(assignment_bytes) > MAX_STATIC_BYTES:
            raise IntegrityError("Assignment vượt giới hạn 5 MiB")
        if len(guideline_bytes) > MAX_STATIC_BYTES:
            raise IntegrityError("Guideline vượt giới hạn 5 MiB")
        self.assignment = assignment
        self.assignment_bytes = assignment_bytes
        self.guideline_bytes = guideline_bytes
        self.suggestions: dict[str, object] | None = None
        self.suggestions_bytes: bytes | None = None
        if suggestions_path is not None:
            suggestions = verify_suggestions(
                load_json(suggestions_path),
                assignment,
            )
            suggestions_bytes = (
                json.dumps(
                    suggestions,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if len(suggestions_bytes) > MAX_STATIC_BYTES:
                raise IntegrityError("Suggestions vượt giới hạn 5 MiB")
            self.suggestions = suggestions
            self.suggestions_bytes = suggestions_bytes
        self.adjudication: dict[str, object] | None = None
        self.adjudication_bytes: bytes | None = None
        if adjudication_path is not None:
            adjudication = verify_adjudication(
                load_json(adjudication_path),
                assignment,
            )
            adjudication_bytes = (
                json.dumps(
                    adjudication,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if len(adjudication_bytes) > MAX_STATIC_BYTES:
                raise IntegrityError("Adjudication input vượt giới hạn 5 MiB")
            self.adjudication = adjudication
            self.adjudication_bytes = adjudication_bytes
        expected_mode = {
            "blind": "BLINDED_INDEPENDENT_ANNOTATION",
            "ai-review": "AI_ASSISTED_HUMAN_VERIFICATION",
            "adjudication": "EXPERT_ADJUDICATION",
        }[workflow_mode]
        if workflow_mode == "blind":
            if (
                self.suggestions is not None
                or self.adjudication is not None
                or assignment["role"] not in {"A", "B"}
            ):
                raise IntegrityError(
                    "Blind mode requires role A/B and forbids "
                    "suggestions/adjudication"
                )
        elif workflow_mode == "ai-review":
            if self.suggestions is None or self.adjudication is not None:
                raise IntegrityError(
                    "AI-review mode requires suggestions and forbids "
                    "adjudication input"
                )
        elif (
            self.adjudication is None
            or self.suggestions is not None
            or assignment["role"] != "ADJUDICATOR"
        ):
            raise IntegrityError(
                "Adjudication mode requires role ADJUDICATOR, a validated "
                "adjudication input and no suggestions"
            )
        workflow_material = "\0".join(
            [
                expected_mode,
                assignment["assignment_id"],
                assignment["assignment_payload_sha256"],
                (
                    self.suggestions["payload_sha256"]
                    if self.suggestions is not None
                    else ""
                ),
                (
                    self.adjudication["payload_sha256"]
                    if self.adjudication is not None
                    else ""
                ),
            ]
        )
        workflow_payload = {
            "workflow_id": (
                "human-absa-workflow-"
                + sha256_text(workflow_material)[:20]
            ),
            "mode": expected_mode,
            "assignment_id": assignment["assignment_id"],
            "assignment_payload_sha256": assignment[
                "assignment_payload_sha256"
            ],
            "suggestions_available": self.suggestions is not None,
            "suggestion_set_id": (
                self.suggestions["payload"]["suggestion_set_id"]
                if self.suggestions is not None
                else None
            ),
            "suggestions_payload_sha256": (
                self.suggestions["payload_sha256"]
                if self.suggestions is not None
                else None
            ),
            "adjudication_available": self.adjudication is not None,
            "adjudication_set_id": (
                self.adjudication["payload"]["adjudication_set_id"]
                if self.adjudication is not None
                else None
            ),
            "adjudication_payload_sha256": (
                self.adjudication["payload_sha256"]
                if self.adjudication is not None
                else None
            ),
        }
        workflow_envelope = {
            "schema_version": "human-absa-workflow/2.0.0",
            "payload": workflow_payload,
            "payload_sha256": sha256_text(
                canonical_json(workflow_payload)
            ),
        }
        self.workflow = workflow_envelope
        self.workflow_bytes = (
            json.dumps(
                workflow_envelope,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        super().__init__(address, AnnotationRequestHandler)

    def handle_error(
        self,
        request: object,
        client_address: tuple[str, int],
    ) -> None:
        error = sys.exc_info()[1]
        if isinstance(
            error,
            (BrokenPipeError, ConnectionAbortedError, ConnectionResetError),
        ):
            return
        super().handle_error(request, client_address)


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: AnnotationServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging opaque annotation IDs or query strings.
        message = format % args
        print(f"[annotation-ui] {self.client_address[0]} {message}")

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        include_body: bool = True,
    ) -> None:
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _route(self, *, include_body: bool) -> None:
        host_header = self.headers.get("Host", "")
        allowed_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host_header.casefold() not in allowed_hosts:
            self._send_bytes(
                b"Invalid Host header\n",
                "text/plain; charset=utf-8",
                status=HTTPStatus.BAD_REQUEST,
                include_body=include_body,
            )
            return
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._send_bytes(
                b'{"status":"ok"}\n',
                "application/json; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/assignment.json":
            self._send_bytes(
                self.server.assignment_bytes,
                "application/json; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/workflow.json":
            self._send_bytes(
                self.server.workflow_bytes,
                "application/json; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/suggestions.json":
            if self.server.suggestions_bytes is None:
                self._send_bytes(
                    b"Not found\n",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                    include_body=include_body,
                )
                return
            self._send_bytes(
                self.server.suggestions_bytes,
                "application/json; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/adjudication.json":
            if self.server.adjudication_bytes is None:
                self._send_bytes(
                    b"Not found\n",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                    include_body=include_body,
                )
                return
            self._send_bytes(
                self.server.adjudication_bytes,
                "application/json; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/guideline.md":
            self._send_bytes(
                self.server.guideline_bytes,
                "text/markdown; charset=utf-8",
                include_body=include_body,
            )
            return
        if path == "/favicon.ico":
            self._send_bytes(
                b"",
                "image/x-icon",
                status=HTTPStatus.NO_CONTENT,
                include_body=False,
            )
            return
        static = STATIC_FILES.get(path)
        if static is None:
            self._send_bytes(
                b"Not found\n",
                "text/plain; charset=utf-8",
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        file_path, content_type = static
        try:
            body = file_path.read_bytes()
        except OSError:
            self._send_bytes(
                b"UI asset missing\n",
                "text/plain; charset=utf-8",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                include_body=include_body,
            )
            return
        if len(body) > MAX_STATIC_BYTES:
            self._send_bytes(
                b"UI asset too large\n",
                "text/plain; charset=utf-8",
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
                include_body=include_body,
            )
            return
        self._send_bytes(body, content_type, include_body=include_body)

    def do_GET(self) -> None:  # noqa: N802
        self._route(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._route(include_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._send_bytes(
            b"Method not allowed\n",
            "text/plain; charset=utf-8",
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one blinded human ABSA assignment on localhost."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("blind", "ai-review", "adjudication"),
        help="Explicit workflow mode; inputs must match or startup fails.",
    )
    parser.add_argument(
        "--assignment",
        type=Path,
        required=True,
        help="Role-specific assignment JSON generated by prepare_reference.",
    )
    parser.add_argument(
        "--guideline",
        type=Path,
        default=None,
        help="Frozen guideline Markdown; defaults to package sibling.",
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=None,
        help=(
            "Validated A/B comparison input; required only in explicit "
            "adjudication mode."
        ),
    )
    parser.add_argument(
        "--suggestions",
        type=Path,
        default=None,
        help=(
            "Optional validated AI suggestions. Supplying this switches the "
            "UI to explicit AI-assisted human-verification mode."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the local UI in the default browser.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assignment_path = args.assignment.resolve()
    guideline_path = (
        args.guideline.resolve()
        if args.guideline is not None
        else assignment_path.parent.parent
        / "ABSA_ANNOTATION_GUIDELINE_V2.md"
    )
    if not assignment_path.is_file():
        raise FileNotFoundError(assignment_path)
    if not guideline_path.is_file():
        raise FileNotFoundError(guideline_path)
    suggestions_path = (
        args.suggestions.resolve()
        if args.suggestions is not None
        else None
    )
    if suggestions_path is not None and not suggestions_path.is_file():
        raise FileNotFoundError(suggestions_path)
    adjudication_path = (
        args.adjudication.resolve()
        if args.adjudication is not None
        else None
    )
    if adjudication_path is not None and not adjudication_path.is_file():
        raise FileNotFoundError(adjudication_path)
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535")

    server = AnnotationServer(
        ("127.0.0.1", args.port),
        assignment_path,
        guideline_path,
        args.mode,
        suggestions_path,
        adjudication_path,
    )
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(
        "Human ABSA UI ready\n"
        f"  Role: {server.assignment['role']}\n"
        f"  Assignment: {server.assignment['assignment_id']}\n"
        f"  Items: {server.assignment['item_count']}\n"
        f"  Mode: {server.workflow['payload']['mode']}\n"
        f"  Workflow: {server.workflow['payload']['workflow_id']}\n"
        f"  URL: {url}\n"
        "  Dữ liệu nháp chỉ nằm trong browser profile này.\n"
        "  Nhấn Ctrl+C để dừng."
    )
    if args.open:
        timer = threading.Timer(0.35, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nĐã dừng annotation UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
