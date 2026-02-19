"""
MangaVID Web Interface

Flask-based web frontend for the MangaVID pipeline.
Upload ZIP, configure voice/description, get loading progress, download video.
"""

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_file, render_template

from pipeline import MangaVideoPipeline
from batch_pipeline import BatchPipeline, BatchResult
from utils.config import Config, load_config

# ── App Setup ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload limit for batch

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# ── Job Tracking ──────────────────────────────────────────────────────────
# In-memory job store  {job_id: {...}}
_jobs: dict = {}
_jobs_lock = threading.Lock()

# Live log buffer per job — keeps last N log lines
_job_logs: dict = {}  # {job_id: [str, ...]}
_MAX_LOG_LINES = 200

STAGE_ORDER = [
    "Input Loading",
    "Panel Detection",
    "Panel Selection",
    "OCR",
    "Story Analysis",
    "Duration Planning",
    "Colorization",
    "Narration",
    "Video Generation",
    "Subtitles",
    "Anime Generation",
]


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _append_log(job_id: str, message: str):
    """Thread-safe append to job log buffer."""
    with _jobs_lock:
        if job_id not in _job_logs:
            _job_logs[job_id] = []
        _job_logs[job_id].append(message)
        if len(_job_logs[job_id]) > _MAX_LOG_LINES:
            _job_logs[job_id] = _job_logs[job_id][-_MAX_LOG_LINES:]


def _run_pipeline(job_id: str, zip_path: Path, config: Config):
    """Worker function – runs in a background thread."""
    try:
        _update_job(job_id, status="running", stage="Initialising pipeline…")

        pipeline = MangaVideoPipeline(config)

        # Monkey-patch the logger so we capture stage transitions
        original_start = pipeline._logger.start_stage
        original_end = pipeline._logger.end_stage
        original_progress = pipeline._logger.progress
        original_info = pipeline._logger.info

        def _hooked_start(stage_name, *a, **kw):
            idx = STAGE_ORDER.index(stage_name) if stage_name in STAGE_ORDER else 0
            pct = int((idx / len(STAGE_ORDER)) * 100)
            _update_job(job_id, stage=stage_name, progress=pct)
            _append_log(job_id, f"[START] {stage_name}")
            return original_start(stage_name, *a, **kw)

        def _hooked_end(stage_name, *a, **kw):
            idx = STAGE_ORDER.index(stage_name) + 1 if stage_name in STAGE_ORDER else 0
            pct = int((idx / len(STAGE_ORDER)) * 100)
            _update_job(job_id, progress=pct)
            _append_log(job_id, f"[DONE] {stage_name}")
            return original_end(stage_name, *a, **kw)

        def _hooked_progress(stage_name, current, total, *a, **kw):
            msg = f"[PROGRESS] {stage_name}: {current}/{total}"
            _append_log(job_id, msg)
            # For OCR, update the sub-status so frontend shows per-panel progress
            if stage_name == "OCR":
                _update_job(job_id, sub_status=f"Panel {current}/{total}")
            elif stage_name == "Anime Generation":
                # Anime sub-progress: map panels into 90-99% range
                pct = 90 + int((current / max(total, 1)) * 9)
                _update_job(job_id, progress=min(pct, 99), sub_status=f"Panel {current}/{total}")
            return original_progress(stage_name, current, total, *a, **kw)

        def _hooked_info(message, *a, **kw):
            _append_log(job_id, message)
            return original_info(message, *a, **kw)

        pipeline._logger.start_stage = _hooked_start
        pipeline._logger.end_stage = _hooked_end
        pipeline._logger.progress = _hooked_progress
        pipeline._logger.info = _hooked_info

        result = pipeline.process(input_path=zip_path)

        if result.success:
            warnings = result.metadata.get('warnings', [])
            
            # Add auto-split suggestion as warning if applicable
            if result.metadata.get('auto_split_suggested'):
                detected = result.metadata.get('total_panels_detected', 0)
                selected = result.metadata.get('panels_selected', 0)
                warnings.append(
                    f"Large chapter detected ({detected} panels, {selected} used). "
                    f"Consider using Batch Mode for better coverage of all content."
                )
            
            _update_job(
                job_id,
                status="done",
                stage="Complete",
                progress=100,
                video_path=str(result.video_path),
                duration=result.duration,
                panel_count=result.panel_count,
                tts_engine=result.metadata.get('tts_engine', ''),
                tts_voice=result.metadata.get('tts_voice', ''),
                tts_fallback=result.metadata.get('tts_fallback', False),
                anime_clip_path=result.metadata.get('anime_clip_path'),
                warnings=warnings,
            )
        else:
            _update_job(job_id, status="error", error=result.error_message or "Unknown error")

    except Exception as exc:
        logger.exception("Pipeline thread crashed")
        _update_job(job_id, status="error", error=str(exc))


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Accept ZIP + settings, kick off pipeline in background, return job_id."""

    # ── File upload ───────────────────────────────────────────────────
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No ZIP file uploaded"}), 400

    # Validate file extension before saving
    allowed_ext = {".zip", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        return jsonify({"error": f"Unsupported file type '{ext}'. Please upload a ZIP or PDF file."}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / file.filename
    file.save(str(zip_path))

    # ── Read form fields ──────────────────────────────────────────────
    description = request.form.get("description", "").strip()
    voice_select = request.form.get("voice_select", "")
    custom_voice_id = request.form.get("custom_voice_id", "").strip()
    tts_engine = request.form.get("tts_engine", "elevenlabs")
    enable_colorization = request.form.get("enable_colorization", "1") == "1"

    # ── Build config ──────────────────────────────────────────────────
    config = load_config()

    # Manga description → anime_title context for Gemini
    if description:
        config.anime_title = description

    # TTS engine
    config.tts_model = tts_engine

    # Colorization
    config.enable_colorization = enable_colorization

    # Anime generation toggle
    enable_anime = request.form.get("enable_anime_gen", "0") == "1"
    config.enable_anime_gen = enable_anime
    anime_style = request.form.get("anime_style", "modern_anime")
    if anime_style:
        config.anime_style = anime_style

    # Voice handling
    if custom_voice_id:
        config.narrator_voice = custom_voice_id
    elif voice_select:
        config.narrator_voice = voice_select

    # ── Register job & launch thread ──────────────────────────────────
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "video_path": None,
            "error": None,
            "warnings": [],
            "tts_engine": "",
            "tts_voice": "",
            "tts_fallback": False,
            "created": datetime.now().isoformat(),
        }

    t = threading.Thread(target=_run_pipeline, args=(job_id, zip_path, config), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    """Poll job progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/logs/<job_id>")
def api_logs(job_id: str):
    """Get live log lines for a job. ?since=N returns only lines after index N."""
    since = int(request.args.get("since", 0))
    with _jobs_lock:
        logs = _job_logs.get(job_id, [])
        lines = logs[since:]
    return jsonify({"lines": lines, "total": since + len(lines)})


@app.route("/api/download/<job_id>")
def api_download(job_id: str):
    """Download the finished video."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Video not ready"}), 400

    video_path = Path(job["video_path"])
    if not video_path.exists():
        return jsonify({"error": "Video file missing"}), 404

    return send_file(
        str(video_path.resolve()),
        as_attachment=True,
        download_name=video_path.name,
        mimetype="video/mp4",
    )


@app.route("/api/download_anime/<job_id>")
def api_download_anime(job_id: str):
    """Download the anime-style clip for a single-mode job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Video not ready"}), 400

    anime_path_str = job.get("anime_clip_path")
    if not anime_path_str:
        return jsonify({"error": "No anime clip was generated"}), 404

    anime_path = Path(anime_path_str)
    if not anime_path.exists():
        return jsonify({"error": "Anime clip file missing"}), 404

    return send_file(
        str(anime_path.resolve()),
        as_attachment=True,
        download_name=anime_path.name,
        mimetype="video/mp4",
    )


# ── Batch Endpoints ──────────────────────────────────────────────────────

def _run_batch_pipeline(job_id: str, zip_path: Path, config: Config):
    """Worker function for batch chapter processing."""
    try:
        _update_job(job_id, status="running", stage="Discovering chapters…")

        batch = BatchPipeline(config)

        # Progress callback for per-chapter updates
        def on_progress(ch_idx, total, stage, pct):
            overall_pct = int((ch_idx / total) * 100)
            _update_job(
                job_id,
                stage=f"{stage} ({ch_idx + 1}/{total})",
                progress=overall_pct,
                batch_current_chapter=ch_idx + 1,
                batch_total_chapters=total,
            )

        batch.set_progress_callback(on_progress)
        result: BatchResult = batch.process_batch(input_path=zip_path)

        if result.success:
            # Build per-chapter video list for frontend
            chapter_videos = []
            for cr in result.chapter_results:
                chapter_videos.append({
                    "chapter": cr.chapter_number,
                    "title": cr.title,
                    "status": cr.status,
                    "video_path": str(cr.video_path) if cr.video_path else None,
                    "duration": cr.duration,
                    "panel_count": cr.panel_count,
                    "error": cr.error,
                })

            _update_job(
                job_id,
                status="done",
                stage="Complete",
                progress=100,
                batch_mode=True,
                chapter_videos=chapter_videos,
                total_chapters=result.total_chapters,
                completed_chapters=result.completed_chapters,
                failed_chapters=result.failed_chapters,
                total_duration=result.total_duration,
                output_directory=str(result.output_directory),
                warnings=result.warnings,
                tts_engine=config.tts_model,
                tts_voice=config.narrator_voice,
            )
        else:
            _update_job(job_id, status="error", error="No chapters could be processed.")

    except Exception as exc:
        logger.exception("Batch pipeline thread crashed")
        _update_job(job_id, status="error", error=str(exc))


@app.route("/api/batch_generate", methods=["POST"])
def api_batch_generate():
    """Accept a ZIP of chapters + settings, kick off batch pipeline."""

    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No ZIP file uploaded"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext != ".zip":
        return jsonify({"error": "Batch mode requires a ZIP file containing chapter folders."}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / file.filename
    file.save(str(zip_path))

    # ── Read form fields ──────────────────────────────────────────────
    description = request.form.get("description", "").strip()
    voice_select = request.form.get("voice_select", "")
    custom_voice_id = request.form.get("custom_voice_id", "").strip()
    tts_engine = request.form.get("tts_engine", "elevenlabs")
    enable_colorization = request.form.get("enable_colorization", "1") == "1"

    config = load_config()
    if description:
        config.anime_title = description
    config.tts_model = tts_engine
    config.enable_colorization = enable_colorization

    # Anime generation toggle
    enable_anime = request.form.get("enable_anime_gen", "0") == "1"
    config.enable_anime_gen = enable_anime
    anime_style = request.form.get("anime_style", "modern_anime")
    if anime_style:
        config.anime_style = anime_style

    if custom_voice_id:
        config.narrator_voice = custom_voice_id
    elif voice_select:
        config.narrator_voice = voice_select

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "batch_mode": True,
            "batch_current_chapter": 0,
            "batch_total_chapters": 0,
            "chapter_videos": [],
            "total_chapters": 0,
            "completed_chapters": 0,
            "failed_chapters": 0,
            "total_duration": 0,
            "output_directory": None,
            "video_path": None,
            "error": None,
            "warnings": [],
            "tts_engine": "",
            "tts_voice": "",
            "tts_fallback": False,
            "created": datetime.now().isoformat(),
        }

    t = threading.Thread(target=_run_batch_pipeline, args=(job_id, zip_path, config), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/download_chapter/<job_id>/<int:chapter_idx>")
def api_download_chapter(job_id: str, chapter_idx: int):
    """Download a specific chapter video from a batch job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Batch not ready"}), 400

    chapters = job.get("chapter_videos", [])
    if chapter_idx < 0 or chapter_idx >= len(chapters):
        return jsonify({"error": "Chapter index out of range"}), 404

    ch = chapters[chapter_idx]
    if ch["status"] != "done" or not ch["video_path"]:
        return jsonify({"error": f"Chapter {ch['chapter']} failed or has no video"}), 400

    video_path = Path(ch["video_path"])
    if not video_path.exists():
        return jsonify({"error": "Video file missing"}), 404

    return send_file(
        str(video_path.resolve()),
        as_attachment=True,
        download_name=video_path.name,
        mimetype="video/mp4",
    )


@app.route("/api/download_batch/<job_id>")
def api_download_batch(job_id: str):
    """Download all chapter videos as a single ZIP."""
    import zipfile as zf

    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Batch not ready"}), 400

    chapters = job.get("chapter_videos", [])
    done_chapters = [c for c in chapters if c["status"] == "done" and c["video_path"]]
    if not done_chapters:
        return jsonify({"error": "No completed videos to download"}), 400

    # Create a temp zip of all chapter videos
    batch_zip_path = UPLOAD_DIR / job_id / "batch_output.zip"
    with zf.ZipFile(str(batch_zip_path), "w", zf.ZIP_STORED) as z:
        for ch in done_chapters:
            vp = Path(ch["video_path"])
            if vp.exists():
                z.write(str(vp), arcname=vp.name)

    return send_file(
        str(batch_zip_path.resolve()),
        as_attachment=True,
        download_name=f"mangavid_batch_{job_id}.zip",
        mimetype="application/zip",
    )


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("MangaVID Web UI → http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
