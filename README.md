# MangaVID

A modular, production-grade Python framework for converting manga files into short-form narrated videos automatically.

## Overview

MangaVID is an intelligent pipeline that accepts manga input (ZIP/PDF/folder) and outputs a short summarized video (<60 seconds) with AI-generated narration. The system acts as an AI editor that understands the story and compresses it into engaging short-form content.

## Features

- **Multi-format Input Support**: ZIP archives, PDF files, or folders of images
- **Intelligent Panel Detection**: Automatically detects and extracts panels from manga pages
- **OCR Text Extraction**: Extracts dialogue and text from panels
- **Story Understanding**: Uses LLM to analyze story and generate narration scripts
- **Smart Panel Selection**: Multiple strategies (LLM, heuristic, hybrid) to select key panels
- **Automatic Colorization**: Optional AI-powered colorization of B&W manga
- **Text-to-Speech Narration**: Multiple TTS backends supported
- **Cinematic Video Generation**: Ken Burns effects, transitions, and subtitles
- **Platform Upload**: Upload to YouTube, TikTok, and more (coming soon)
- **Fully Configurable**: All parameters controllable via config.json

## Installation

### Prerequisites

1. **Python 3.9+** - Required
2. **FFmpeg** - Required for video encoding

   ```bash
   # Windows (using chocolatey)
   choco install ffmpeg
   
   # Mac
   brew install ffmpeg
   
   # Linux
   sudo apt-get install ffmpeg
   ```

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd MangaVID

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install core dependencies
pip install numpy opencv-python Pillow pyttsx3

# (Optional) Install additional features
pip install easyocr  # Better OCR for manga
pip install openai   # LLM-powered story intelligence
```

## Usage

### Basic Usage

```bash
# Process a ZIP file
python main.py ./my_manga.zip

# Process a folder of images
python main.py ./manga_pages/

# Process a PDF
python main.py ./manga.pdf

# Specify output path
python main.py ./manga.zip --output ./videos/my_video.mp4
```

### Command Line Options

```bash
python main.py <input> [options]

Options:
  -o, --output PATH       Output video path
  -c, --config PATH       Configuration file path
  -t, --title TEXT        Video title
  --max-duration SECONDS  Maximum video duration (default: 60)
  --panel-mode MODE       Panel selection mode: llm, heuristic, hybrid
  --max-panels N          Maximum panels to include
  --no-colorize          Disable colorization
  --no-subtitles         Disable subtitles
  --llm-model MODEL      LLM model (placeholder, gpt-4, etc.)
  --tts-model MODEL      TTS model (pyttsx3, gtts, placeholder)
  -v, --verbose          Enable debug logging
  --generate-config      Generate default config.json
```

### Python API

```python
from pipeline import MangaVideoPipeline, create_pipeline
from utils.config import Config

# Using default configuration
pipeline = create_pipeline()
result = pipeline.process("./manga.zip")

if result.success:
    print(f"Video created: {result.video_path}")
    print(f"Duration: {result.duration}s")

# With custom configuration
config = Config(
    max_video_duration=45,
    panel_selection_mode="hybrid",
    llm_model="gpt-4",
    tts_model="gtts"
)
pipeline = MangaVideoPipeline(config)
result = pipeline.process("./manga.zip", output_path="./output.mp4")
```

## Configuration

All settings are controlled via `config.json`:

```json
{
  "max_video_duration": 60,
  "video_width": 1080,
  "video_height": 1920,
  "video_fps": 30,
  
  "narration_style": "engaging",
  "tts_model": "pyttsx3",
  
  "panel_selection_mode": "hybrid",
  "max_panels": 10,
  
  "llm_model": "placeholder",
  "colorization_model": "placeholder",
  
  "enable_colorization": true,
  "enable_subtitles": true,
  "enable_transitions": true
}
```

### Key Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_video_duration` | Target video length in seconds | 60 |
| `panel_selection_mode` | `llm`, `heuristic`, or `hybrid` | `hybrid` |
| `llm_model` | LLM for story analysis | `placeholder` |
| `tts_model` | TTS engine | `pyttsx3` |
| `colorization_model` | Colorization model | `placeholder` |
| `reading_direction` | `rtl` for manga, `ltr` for comics | `rtl` |

## Architecture

```
MangaVID/
├── main.py              # CLI entry point
├── pipeline.py          # Main pipeline orchestrator
├── config.json          # Configuration file
│
├── interfaces/          # Abstract base classes
│   ├── base_input_handler.py
│   ├── base_panel_detector.py
│   ├── base_ocr_engine.py
│   ├── base_story_intelligence.py
│   ├── base_panel_selector.py
│   ├── base_colorizer.py
│   ├── base_narrator.py
│   ├── base_video_generator.py
│   └── base_uploader.py
│
├── modules/             # Concrete implementations
│   ├── input_handler.py
│   ├── panel_detector.py
│   ├── ocr_engine.py
│   ├── story_intelligence.py
│   ├── panel_selector.py
│   ├── colorizer.py
│   ├── narrator.py
│   ├── video_generator.py
│   └── uploader.py
│
├── utils/               # Utilities
│   ├── config.py
│   ├── logger.py
│   └── duration_controller.py
│
└── outputs/             # Generated videos
```

## Pipeline Flow

```
Input (ZIP/PDF/Folder)
        │
        ▼
┌─────────────────┐
│  Input Handler  │ → Load and validate manga pages
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Panel Detector  │ → Detect panels, determine reading order
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   OCR Engine    │ → Extract text from panels
└─────────────────┘
        │
        ▼
┌─────────────────┐
│Story Intelligence│ → Generate narration script, select key panels
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Panel Selector  │ → Finalize panel selection using configured strategy
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   Colorizer     │ → Optionally colorize B&W panels
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    Narrator     │ → Generate audio narration (TTS)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│Video Generator  │ → Create video with transitions, effects, subtitles
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    Uploader     │ → Upload to platform (optional)
└─────────────────┘
        │
        ▼
    Output Video
```

## Extending the Framework

### Replacing AI Models

All AI-powered modules inherit from abstract interfaces, making them swappable:

#### Custom OCR Engine

```python
from interfaces.base_ocr_engine import BaseOCREngine, OCRResult

class MyCustomOCR(BaseOCREngine):
    def extract_text(self, panel_image, panel_index):
        # Your implementation
        return OCRResult(...)
    
    # Implement other required methods...
```

#### Custom Story Intelligence

```python
from interfaces.base_story_intelligence import BaseStoryIntelligence

class MyStoryEngine(BaseStoryIntelligence):
    def analyze(self, input_data):
        # Use your preferred LLM
        return StoryIntelligenceOutput(...)
```

#### Custom TTS

```python
from interfaces.base_narrator import BaseNarrator

class MyTTSNarrator(BaseNarrator):
    def generate(self, script, config):
        # Use your preferred TTS service
        return NarrationResult(...)
```

### Adding Custom Panel Scoring

```python
from modules.panel_selector import PanelSelector

selector = PanelSelector()

# Register a custom scoring function
def my_scoring(panel, ocr_result):
    # Your scoring logic
    return score_between_0_and_1

selector.register_scoring_function(
    name="my_custom_scorer",
    func=my_scoring,
    weight=0.3
)
```

## Story Intelligence Output Format

The Story Intelligence module outputs JSON in this format:

```json
{
    "summary_script": "The narration text for the video...",
    "selected_panels": [0, 2, 5, 8, 12],
    "tone": "dramatic",
    "key_events": ["Hero awakens", "Battle begins", "Victory achieved"],
    "characters": ["Protagonist", "Antagonist"]
}
```

## Supported Platforms

### Input Formats
- ZIP archives containing images
- PDF files (requires poppler)
- Folders with JPG, PNG, WEBP, BMP, GIF

### Output Formats
- MP4 video (H.264, AAC audio)
- Vertical format optimized for short-form (1080x1920)

### Upload Destinations
- Local filesystem
- YouTube / YouTube Shorts (requires API setup)
- TikTok, Instagram Reels (planned)

## Troubleshooting

### Common Issues

**"No OCR backend available"**
- Install easyocr: `pip install easyocr`
- Or install pytesseract: `pip install pytesseract` and [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your PATH
- Test with: `ffmpeg -version`

**"PDF support not available"**
- Install pdf2image: `pip install pdf2image`
- Install poppler: See [poppler installation guide](https://github.com/Belval/pdf2image#how-to-install)

**Video has no audio**
- Ensure TTS model is working: `pip install pyttsx3`
- Check for TTS errors in logs

### Performance Tips

- Use GPU if available for OCR/colorization
- Reduce `video_fps` for faster processing
- Use `placeholder` models for testing
- Process fewer panels with `--max-panels`

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- OpenCV for image processing
- FFmpeg for video encoding
- EasyOCR for text extraction
- All the manga creators whose work inspires this project
#   M a n g a A n i m a t o r  
 