#!/usr/bin/env python3
"""
MangaVID - Manga to Video Converter

A modular, production-grade framework for converting manga files
into short-form narrated videos.

Usage:
    python main.py <input_path> [options]
    
    python main.py ./manga.zip
    python main.py ./manga_folder --output ./video.mp4
    python main.py ./manga.pdf --config ./my_config.json

Examples:
    # Process a ZIP archive  
    python main.py "./my_manga.zip"
    
    # Process a folder with custom output path
    python main.py "./manga_pages" --output "./output/my_video.mp4"
    
    # Use custom configuration
    python main.py "./manga.zip" --config "./custom_config.json"
    
    # Override specific settings
    python main.py "./manga.zip" --max-duration 45 --panel-mode hybrid
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline import MangaVideoPipeline, create_pipeline
from utils.config import Config, load_config, save_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MangaVID - Convert manga to narrated videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Path to manga file (ZIP/PDF) or folder containing images"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output video path (default: auto-generated in outputs/)"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "-t", "--title",
        type=str,
        default=None,
        help="Title for the video"
    )
    
    # Override arguments
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum video duration in seconds"
    )
    
    parser.add_argument(
        "--panel-mode",
        choices=["llm", "heuristic", "hybrid"],
        default=None,
        help="Panel selection mode"
    )
    
    parser.add_argument(
        "--max-panels",
        type=int,
        default=None,
        help="Maximum number of panels to include"
    )
    
    parser.add_argument(
        "--no-colorize",
        action="store_true",
        help="Disable colorization"
    )
    
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Disable subtitles"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model for story intelligence (e.g., gpt-4, placeholder)"
    )
    
    parser.add_argument(
        "--tts-model",
        type=str,
        default=None,
        help="TTS model for narration (e.g., pyttsx3, gtts, placeholder)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a default config.json file and exit"
    )
    
    return parser.parse_args()


def apply_overrides(config: Config, args) -> Config:
    """Apply command-line overrides to configuration."""
    if args.max_duration is not None:
        config.max_video_duration = args.max_duration
    
    if args.panel_mode is not None:
        config.panel_selection_mode = args.panel_mode
    
    if args.max_panels is not None:
        config.max_panels = args.max_panels
    
    if args.no_colorize:
        config.enable_colorization = False
    
    if args.no_subtitles:
        config.enable_subtitles = False
    
    if args.llm_model is not None:
        config.llm_model = args.llm_model
    
    if args.tts_model is not None:
        config.tts_model = args.tts_model
    
    if args.verbose:
        config.log_level = "DEBUG"
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle config generation
    if args.generate_config:
        default_config = Config()
        save_config(default_config, Path("config.json"))
        print("Generated default config.json")
        return 0
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        return 1
    
    # Load configuration
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    # Create and run pipeline
    print(f"MangaVID - Processing: {args.input}")
    print("-" * 50)
    
    try:
        pipeline = MangaVideoPipeline(config)
        result = pipeline.process(
            input_path=args.input,
            output_path=args.output,
            title=args.title
        )
        
        if result.success:
            print("-" * 50)
            print("SUCCESS!")
            print(f"  Video: {result.video_path}")
            print(f"  Duration: {result.duration:.1f} seconds")
            print(f"  Panels: {result.panel_count}")
            print(f"  Processing time: {result.metadata.get('total_processing_time', 0):.1f}s")
            return 0
        else:
            print("-" * 50)
            print(f"FAILED: {result.error_message}", file=sys.stderr)
            return 1
            
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
