"""
Smart CCTV Surveillance System - FastAPI Server & Web Dashboard
================================================================
Handles multi-camera streaming, real-time alerting, video file uploads,
and timestamped incident logging.
"""

import os
import sys
import csv
import io
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dashboard.camera_manager import MultiCameraManager, UPLOADS_DIR


# Global multi-camera manager
camera_manager = MultiCameraManager()


def init_default_cameras():
    """Initializes default surveillance cameras so dashboard has immediate feeds."""
    sample_burglary = BASE_DIR / "datasets" / "raw" / "theft_intrusion" / "burglary" / "Burglary001_x264.mp4"
    sample_fight = BASE_DIR / "datasets" / "raw" / "violence" / "fights" / "Fighting002_x264.mp4"

    if sample_burglary.exists():
        camera_manager.add_camera("cam_1", "Perimeter South (Recorded)", str(sample_burglary), mode="recorded")
    if sample_fight.exists():
        camera_manager.add_camera("cam_2", "Lobby Hallway (Recorded)", str(sample_fight), mode="recorded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_default_cameras()
    yield
    # Cleanup camera streams on shutdown
    for cam in list(camera_manager.cameras.values()):
        cam.stop()


app = FastAPI(title="Smart CCTV Surveillance Operations Center", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "dashboard" / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serves the main CCTV Security Dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"cameras": list(camera_manager.cameras.values())}
    )


def gen_frames(camera_id: str):
    """Generator function for MJPEG video streaming."""
    cam = camera_manager.cameras.get(camera_id)
    if not cam:
        return
    while cam.is_running:
        frame_bytes = cam.get_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            import time
            time.sleep(0.04)


@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    """Endpoint for continuous real-time camera stream."""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail="Camera feed not found")
    return StreamingResponse(gen_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/add_camera")
def add_camera(
    camera_id: str = Form(...),
    name: str = Form(...),
    source: str = Form(...),
    mode: str = Form("live")
):
    """Adds a new live webcam, RTSP stream, or local video path."""
    cam = camera_manager.add_camera(camera_id, name, source, mode=mode)
    if cam:
        return {"status": "success", "message": f"Camera '{name}' started in {mode} mode"}
    raise HTTPException(status_code=400, detail="Failed to open camera source")


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), name: str = Form("Uploaded Surveillance Video")):
    """Uploads a user video file and immediately creates a RECORDED analysis feed."""
    dest_path = UPLOADS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cam_id = f"upload_{int(Path(file.filename).stem[:8] if len(Path(file.filename).stem) >= 8 else 'custom')}_{len(camera_manager.cameras)+1}"
    cam = camera_manager.add_camera(cam_id, name, str(dest_path), mode="recorded")
    if cam:
        return {"status": "success", "camera_id": cam_id, "name": name, "mode": "recorded"}
    raise HTTPException(status_code=400, detail="Failed to process uploaded video")


@app.delete("/api/remove_camera/{camera_id}")
def remove_camera(camera_id: str):
    """Stops and removes a camera feed."""
    camera_manager.remove_camera(camera_id)
    return {"status": "success", "message": f"Removed camera {camera_id}"}


@app.get("/api/status")
def get_status():
    """Returns active cameras and active LIVE threats (for siren triggering)."""
    cams = []
    for c in camera_manager.cameras.values():
        cams.append({
            "id": c.camera_id,
            "name": c.name,
            "mode": c.mode,
            "is_running": c.is_running,
            "active_threats": list(c.active_threats)
        })

    live_threats = camera_manager.get_active_live_threats()
    return {
        "cameras": cams,
        "live_siren_alert": len(live_threats) > 0,
        "live_threats": [t.upper() for t in live_threats]
    }


@app.get("/api/incidents")
def get_incidents():
    """Returns all timestamped security incidents recorded across all cameras."""
    incidents = camera_manager.get_all_incidents()
    return {"incidents": incidents, "total": len(incidents)}


@app.get("/api/export_report")
def export_report():
    """Generates a downloadable CSV Incident Summary Report."""
    incidents = camera_manager.get_all_incidents()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Camera Name", "Feed Mode", "Threat Type", "Video Timecode"])

    for inc in incidents:
        writer.writerow([inc["real_time"], inc["camera_name"], inc["mode"].upper(), inc["threat"], inc["video_time"]])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cctv_security_incident_report.csv"}
    )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print(" SMART CCTV SURVEILLANCE OPERATIONS CENTER")
    print(" Dashboard running at: http://127.0.0.1:8000")
    print("=" * 70 + "\n")
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=False)
