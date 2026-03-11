"""
server.py — Streaming MJPEG + API REST de eventos (Flask)
Acesse: http://<ip>:5000
"""

import cv2
import time
import threading
import queue
from flask import Flask, Response, render_template, jsonify

from config import STREAM_HOST, STREAM_PORT, STREAM_JPEG_QUALITY, MAX_EVENTS

app = Flask(__name__)

# ── Estado global (thread-safe) ──────────────────────────────────
_frame_lock   = threading.Lock()
_latest_frame: bytes = b""
_events_lock  = threading.Lock()
_events: list  = []
_stats = {
    "fps": 0.0,
    "total_interactions": 0,
    "uptime_start": time.time(),
}


def push_frame(frame_bgr):
    """Chamado pelo loop principal a cada frame processado."""
    global _latest_frame
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    if ok:
        with _frame_lock:
            _latest_frame = buf.tobytes()


def push_event(event):
    """Chamado pelo detector quando uma interação é confirmada."""
    with _events_lock:
        _events.insert(0, {
            "time":    event.time_str,
            "product": event.product_label,
            "frames":  event.frame_count,
        })
        if len(_events) > MAX_EVENTS:
            _events.pop()
    _stats["total_interactions"] += 1


def update_fps(fps: float):
    _stats["fps"] = round(fps, 1)


# ── Rotas ─────────────────────────────────────────────────────────

def _gen_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events")
def api_events():
    with _events_lock:
        evs = list(_events)
    return jsonify(evs)


@app.route("/api/stats")
def api_stats():
    uptime = int(time.time() - _stats["uptime_start"])
    h, m, s = uptime//3600, (uptime%3600)//60, uptime%60
    return jsonify({
        **_stats,
        "uptime": f"{h:02d}:{m:02d}:{s:02d}",
    })


def start(debug=False):
    app.run(host=STREAM_HOST, port=STREAM_PORT,
            debug=debug, threaded=True, use_reloader=False)
