"""
MangaVID Batch Pipeline

Processes multiple manga chapters as a connected series, generating
individual videos with inter-chapter continuity, cliffhangers, and
optional filler/bridge segments.
"""

import json
import logging
import shutil
import zipfile
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from pipeline import MangaVideoPipeline, PipelineResult
from utils.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """Metadata for a single chapter within a batch."""
    chapter_number: int
    source_path: Path
    title: str = ""
    order_index: int = 0  # position in the batch


@dataclass
class BatchChapterResult:
    """Result for a single chapter within a batch run."""
    chapter_number: int
    title: str
    pipeline_result: Optional[PipelineResult] = None
    video_path: Optional[Path] = None
    duration: float = 0.0
    panel_count: int = 0
    status: str = "pending"  # pending | running | done | failed | skipped
    error: Optional[str] = None
    narration_script: str = ""
    tone: str = ""
    characters: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of a full batch pipeline run."""
    success: bool
    total_chapters: int = 0
    completed_chapters: int = 0
    failed_chapters: int = 0
    skipped_chapters: int = 0
    chapter_results: List[BatchChapterResult] = field(default_factory=list)
    total_duration: float = 0.0
    total_processing_time: float = 0.0
    output_directory: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_chapters": self.total_chapters,
            "completed_chapters": self.completed_chapters,
            "failed_chapters": self.failed_chapters,
            "skipped_chapters": self.skipped_chapters,
            "total_duration": self.total_duration,
            "total_processing_time": self.total_processing_time,
            "output_directory": str(self.output_directory) if self.output_directory else None,
            "warnings": self.warnings,
            "chapters": [
                {
                    "chapter_number": cr.chapter_number,
                    "title": cr.title,
                    "status": cr.status,
                    "video_path": str(cr.video_path) if cr.video_path else None,
                    "duration": cr.duration,
                    "panel_count": cr.panel_count,
                    "error": cr.error,
                }
                for cr in self.chapter_results
            ],
            "metadata": self.metadata,
        }


class BatchPipeline:
    """
    Orchestrates processing of multiple manga chapters as a connected series.

    Features:
    - Sorts uploaded chapters by number automatically
    - Passes previous-chapter context to story AI for continuity
    - Injects cliffhanger endings via prompt engineering
    - Adds stage-setting intro panels when the AI deems it appropriate
    - Skips failed chapters and continues (resilience)
    - Reports per-chapter status for frontend tracking
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback for progress updates: callback(chapter_idx, total, stage, pct)."""
        self._progress_callback = callback

    def _emit_progress(self, chapter_idx: int, total: int, stage: str, pct: int):
        if self._progress_callback:
            try:
                self._progress_callback(chapter_idx, total, stage, pct)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def process_batch(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
    ) -> BatchResult:
        """
        Process a batch of manga chapters.

        Args:
            input_path: A ZIP containing per-chapter sub-folders or ZIPs,
                        OR a directory of chapter folders / chapter ZIPs.
            output_dir: Directory for output videos (auto-generated if None).

        Returns:
            BatchResult with per-chapter results.
        """
        start = datetime.now()
        warnings: List[str] = []

        # ── 1. Discover and sort chapters ────────────────────────
        chapters = self._discover_chapters(input_path)
        if not chapters:
            return BatchResult(
                success=False,
                warnings=["No chapters found in the uploaded file."],
            )

        total = len(chapters)
        logger.info(f"Batch pipeline: discovered {total} chapters")

        # ── 2. Prepare output directory ──────────────────────────
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_directory) / f"batch_{ts}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 3. Process each chapter ──────────────────────────────
        chapter_results: List[BatchChapterResult] = []
        previous_context = _empty_context()
        completed = 0
        failed = 0
        skipped = 0

        for idx, chapter in enumerate(chapters):
            cr = BatchChapterResult(
                chapter_number=chapter.chapter_number,
                title=chapter.title,
                status="running",
            )
            self._emit_progress(idx, total, f"Chapter {chapter.chapter_number}", int(idx / total * 100))

            try:
                result = self._process_single_chapter(
                    chapter=chapter,
                    output_dir=output_dir,
                    previous_context=previous_context,
                    chapter_index=idx,
                    total_chapters=total,
                )

                if result.success:
                    cr.status = "done"
                    cr.pipeline_result = result
                    cr.video_path = result.video_path
                    cr.duration = result.duration
                    cr.panel_count = result.panel_count
                    cr.narration_script = result.narration_script
                    cr.tone = result.metadata.get("story_tone", "")
                    cr.characters = result.metadata.get("characters", [])
                    cr.key_events = result.metadata.get("key_events", [])
                    completed += 1

                    # Build context for the *next* chapter
                    previous_context = {
                        "previous_narration": result.narration_script,
                        "previous_tone": cr.tone,
                        "previous_characters": cr.characters,
                        "previous_key_events": cr.key_events,
                        "previous_chapter": chapter.chapter_number,
                    }

                    # Capture per-chapter warnings
                    ch_warnings = result.metadata.get("warnings", [])
                    for w in ch_warnings:
                        warnings.append(f"Ch {chapter.chapter_number}: {w}")
                else:
                    cr.status = "failed"
                    cr.error = result.error_message or "Unknown error"
                    failed += 1
                    warnings.append(f"Ch {chapter.chapter_number} failed: {cr.error}")
                    # Keep previous_context unchanged so the *next* chapter can still reference older context

            except Exception as exc:
                logger.exception(f"Chapter {chapter.chapter_number} crashed")
                cr.status = "failed"
                cr.error = str(exc)
                failed += 1
                warnings.append(f"Ch {chapter.chapter_number} crashed: {exc}")

            chapter_results.append(cr)

        total_time = (datetime.now() - start).total_seconds()
        total_duration = sum(cr.duration for cr in chapter_results)

        return BatchResult(
            success=completed > 0,
            total_chapters=total,
            completed_chapters=completed,
            failed_chapters=failed,
            skipped_chapters=skipped,
            chapter_results=chapter_results,
            total_duration=total_duration,
            total_processing_time=total_time,
            output_directory=output_dir,
            warnings=warnings,
            metadata={
                "config_anime_title": self.config.anime_title,
                "total_processing_time": total_time,
            },
        )

    # ──────────────────────────────────────────────────────────────
    # Chapter discovery
    # ──────────────────────────────────────────────────────────────

    def _discover_chapters(self, input_path: Path) -> List[ChapterInfo]:
        """
        Discover chapters from the input.

        Supports:
        1. A single ZIP containing sub-folders named like ch001/, chapter_1/, etc.
        2. A single ZIP containing inner ZIPs (ch001.zip, ch002.zip, …).
        3. A directory of chapter sub-folders.
        4. A directory of chapter ZIPs.
        """
        input_path = Path(input_path)
        chapters: List[ChapterInfo] = []

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            chapters = self._discover_from_zip(input_path)
        elif input_path.is_dir():
            chapters = self._discover_from_directory(input_path)
        else:
            logger.error(f"Unsupported batch input: {input_path}")

        # Sort by chapter number
        chapters.sort(key=lambda c: c.chapter_number)

        # Assign order_index
        for i, ch in enumerate(chapters):
            ch.order_index = i

        return chapters

    def _discover_from_zip(self, zip_path: Path) -> List[ChapterInfo]:
        """Discover chapters packed inside a single ZIP."""
        chapters: List[ChapterInfo] = []
        extract_dir = zip_path.parent / f"_batch_extract_{zip_path.stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Now treat extracted directory as a chapter source
        chapters = self._discover_from_directory(extract_dir)

        # If no sub-folders found, check for inner ZIPs
        if not chapters:
            for inner_zip in sorted(extract_dir.glob("*.zip")):
                num = self._extract_chapter_number(inner_zip.stem)
                chapters.append(ChapterInfo(
                    chapter_number=num,
                    source_path=inner_zip,
                    title=f"Chapter {num}",
                ))

        return chapters

    def _discover_from_directory(self, dir_path: Path) -> List[ChapterInfo]:
        """Discover chapters from a directory of sub-folders or ZIPs."""
        chapters: List[ChapterInfo] = []
        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        # Check sub-folders that contain images
        for sub in sorted(dir_path.iterdir()):
            if sub.name.startswith("_") or sub.name.startswith("."):
                continue
            if sub.is_dir():
                has_images = any(
                    f.suffix.lower() in image_exts
                    for f in sub.iterdir()
                    if f.is_file()
                )
                if has_images:
                    num = self._extract_chapter_number(sub.name)
                    chapters.append(ChapterInfo(
                        chapter_number=num,
                        source_path=sub,
                        title=f"Chapter {num}",
                    ))
            elif sub.is_file() and sub.suffix.lower() == ".zip":
                num = self._extract_chapter_number(sub.stem)
                chapters.append(ChapterInfo(
                    chapter_number=num,
                    source_path=sub,
                    title=f"Chapter {num}",
                ))

        return chapters

    @staticmethod
    def _extract_chapter_number(name: str) -> int:
        """Extract chapter number from a folder/file name.

        Handles: ch001, chapter_1, Chapter 100, vol2_ch15, 003, etc.
        """
        # Try common patterns
        patterns = [
            r'ch(?:apter)?[_\s.-]*(\d+)',  # ch001, chapter_1, chapter 100
            r'(\d+)',                        # bare number
        ]
        for pat in patterns:
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                return int(m.group(1))
        # Fallback: hash the name to get a stable ordering
        return abs(hash(name)) % 100000

    # ──────────────────────────────────────────────────────────────
    # Single-chapter processing (with batch context)
    # ──────────────────────────────────────────────────────────────

    def _process_single_chapter(
        self,
        chapter: ChapterInfo,
        output_dir: Path,
        previous_context: Dict[str, Any],
        chapter_index: int,
        total_chapters: int,
    ) -> PipelineResult:
        """Process one chapter with batch-aware context injection."""
        logger.info(f"Processing chapter {chapter.chapter_number} ({chapter_index + 1}/{total_chapters})")

        # Build a per-chapter config clone
        config = self._clone_config(self.config)

        # Inject batch context into anime_title so the story AI sees it
        batch_context = self._build_batch_context(
            chapter=chapter,
            previous_context=previous_context,
            chapter_index=chapter_index,
            total_chapters=total_chapters,
        )
        config.anime_title = batch_context

        # Create and run the pipeline
        pipeline = MangaVideoPipeline(config)

        # Output path for this chapter
        safe_name = re.sub(r'[^\w\-.]', '_', f"ch{chapter.chapter_number:04d}")
        output_path = output_dir / f"{safe_name}.mp4"

        result = pipeline.process(
            input_path=chapter.source_path,
            output_path=output_path,
            title=f"Chapter {chapter.chapter_number}",
        )

        return result

    def _build_batch_context(
        self,
        chapter: ChapterInfo,
        previous_context: Dict[str, Any],
        chapter_index: int,
        total_chapters: int,
    ) -> str:
        """
        Build the anime_title / context string that gets passed to Story Intelligence.

        This injects:
        - Series name & chapter number
        - Previous chapter summary (for continuity)
        - Cliffhanger instruction (for all but the last chapter)
        - Stage-setting instruction (for the first chapter or when context changes)
        """
        parts = []

        # Base anime title
        base_title = self.config.anime_title or "Manga Series"
        parts.append(f"{base_title} — Chapter {chapter.chapter_number}")

        # ── Previous-chapter recap (continuity) ──
        prev_narration = previous_context.get("previous_narration", "")
        prev_events = previous_context.get("previous_key_events", [])
        prev_chars = previous_context.get("previous_characters", [])
        prev_chapter = previous_context.get("previous_chapter")

        if prev_narration:
            recap_lines = []
            recap_lines.append(f"\n[PREVIOUS CHAPTER CONTEXT — Chapter {prev_chapter}]")
            # Truncate to ~80 words for token efficiency
            recap_words = prev_narration.split()[:80]
            recap_lines.append(f"Previous narration (summary): {' '.join(recap_words)}")
            if prev_events:
                recap_lines.append(f"Key events: {', '.join(prev_events[:5])}")
            if prev_chars:
                recap_lines.append(f"Active characters: {', '.join(prev_chars[:6])}")
            recap_lines.append(
                "Use this context for smooth continuation. Do NOT repeat what was already said. "
                "The viewer has seen the previous video. Start where it left off."
            )
            parts.append("\n".join(recap_lines))

        # ── Cliffhanger instruction ──
        if chapter_index < total_chapters - 1:
            parts.append(
                "\n[CLIFFHANGER ENDING — MANDATORY]\n"
                "This chapter is part of a multi-part series. "
                "End the narration on a dramatic hook that makes the viewer NEED to watch the next part. "
                "Use an open question, an unresolved threat, or a shocking reveal. "
                "Do NOT wrap up the story neatly — leave the viewer hanging."
            )
        else:
            parts.append(
                "\n[SERIES FINALE]\n"
                "This is the last chapter in this batch. Give it a satisfying conclusion "
                "while still leaving room for future episodes."
            )

        # ── Stage-setting instruction ──
        if chapter_index == 0 or not prev_narration:
            parts.append(
                "\n[STAGE-SETTING — FIRST EPISODE]\n"
                "This is the opening episode. Use the strongest, most visually striking panel "
                "to SET THE STAGE. If there's a powerful character introduction or a dramatic "
                "wide shot, use it as the opening visual with narration that poses a compelling "
                "question or sets the tone. Example: 'Why is [character] considered the strongest?' "
                "over a wide shot of the character. Only do this if the visuals support it — "
                "never force it."
            )
        else:
            parts.append(
                "\n[STAGE-SETTING — CONTINUATION]\n"
                "If the story's setting or situation has dramatically changed from the previous "
                "chapter, pick ONE strong panel to re-establish the scene. A dramatic location "
                "change, a new villain reveal, or a time skip deserves a stage-setting moment. "
                "If nothing has changed dramatically, skip stage-setting and jump straight in."
            )

        return "\n".join(parts)

    @staticmethod
    def _clone_config(config: Config) -> Config:
        """Create a shallow clone of a Config object."""
        return Config.from_dict(config.to_dict())


def _empty_context() -> Dict[str, Any]:
    """Return an empty previous-chapter context."""
    return {
        "previous_narration": "",
        "previous_tone": "",
        "previous_characters": [],
        "previous_key_events": [],
        "previous_chapter": None,
    }
