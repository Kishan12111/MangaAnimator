"""
Input Handler Module

Handles loading manga from various formats: ZIP, PDF, and folders.
"""

import logging
import zipfile
from pathlib import Path
from typing import List, Optional
import io

import numpy as np
from PIL import Image

from interfaces.base_input_handler import BaseInputHandler, InputResult, MangaPage

logger = logging.getLogger(__name__)


class InputHandler(BaseInputHandler):
    """
    Concrete implementation of manga input handling.
    
    Supports ZIP archives, PDF files, and image folders.
    """
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    def __init__(self):
        self._pdf_available = self._check_pdf_support()
    
    def _check_pdf_support(self) -> bool:
        """Check if PDF support is available."""
        try:
            import pdf2image
            return True
        except ImportError:
            logger.warning("pdf2image not installed. PDF support disabled.")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats."""
        formats = ['zip', 'folder']
        if self._pdf_available:
            formats.append('pdf')
        return formats
    
    def validate(self, input_path: Path) -> bool:
        """Validate if the input can be processed."""
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return False
        
        if input_path.is_dir():
            # Check if folder contains images
            images = self._get_images_from_folder(input_path)
            return len(images) > 0
        
        suffix = input_path.suffix.lower()
        
        if suffix == '.zip':
            try:
                with zipfile.ZipFile(input_path, 'r') as zf:
                    # Check for image files in zip
                    for name in zf.namelist():
                        if Path(name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                            return True
                return False
            except zipfile.BadZipFile:
                logger.error(f"Invalid ZIP file: {input_path}")
                return False
        
        if suffix == '.pdf':
            if not self._pdf_available:
                logger.error("PDF support not available")
                return False
            return True
        
        logger.error(f"Unsupported format: {suffix}")
        return False
    
    def load(self, input_path: Path) -> InputResult:
        """Load manga from the given input path."""
        if not self.validate(input_path):
            raise ValueError(f"Invalid or unsupported input: {input_path}")
        
        input_path = Path(input_path)
        
        if input_path.is_dir():
            return self._load_from_folder(input_path)
        
        suffix = input_path.suffix.lower()
        
        if suffix == '.zip':
            return self._load_from_zip(input_path)
        elif suffix == '.pdf':
            return self._load_from_pdf(input_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _get_images_from_folder(self, folder_path: Path) -> List[Path]:
        """Get sorted list of image files from a folder."""
        images = []
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            images.extend(folder_path.glob(f'*{ext}'))
            images.extend(folder_path.glob(f'*{ext.upper()}'))
        
        # Sort by filename for proper page ordering
        images.sort(key=lambda x: x.name.lower())
        return images
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load an image file as numpy array."""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def _load_image_from_bytes(self, data: bytes) -> np.ndarray:
        """Load an image from bytes as numpy array."""
        img = Image.open(io.BytesIO(data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    
    def _load_from_folder(self, folder_path: Path) -> InputResult:
        """Load manga from a folder of images."""
        logger.info(f"Loading manga from folder: {folder_path}")
        
        image_paths = self._get_images_from_folder(folder_path)
        pages = []
        
        for idx, img_path in enumerate(image_paths):
            try:
                image = self._load_image(img_path)
                page = MangaPage(
                    index=idx,
                    image=image,
                    source_path=img_path,
                    metadata={'filename': img_path.name}
                )
                pages.append(page)
                logger.debug(f"Loaded page {idx}: {img_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
        
        logger.info(f"Loaded {len(pages)} pages from folder")
        
        return InputResult(
            pages=pages,
            source_type='folder',
            total_pages=len(pages),
            metadata={
                'source_path': str(folder_path),
                'image_count': len(image_paths)
            }
        )
    
    def _load_from_zip(self, zip_path: Path) -> InputResult:
        """Load manga from a ZIP archive."""
        logger.info(f"Loading manga from ZIP: {zip_path}")
        
        pages = []
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all image files and sort them
            image_names = [
                name for name in zf.namelist()
                if Path(name).suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS
                and not name.startswith('__MACOSX')  # Skip Mac metadata
            ]
            image_names.sort()
            
            for idx, name in enumerate(image_names):
                try:
                    data = zf.read(name)
                    image = self._load_image_from_bytes(data)
                    page = MangaPage(
                        index=idx,
                        image=image,
                        source_path=Path(name),
                        metadata={'filename': Path(name).name, 'archive_path': name}
                    )
                    pages.append(page)
                    logger.debug(f"Loaded page {idx}: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load image from ZIP {name}: {e}")
        
        logger.info(f"Loaded {len(pages)} pages from ZIP")
        
        return InputResult(
            pages=pages,
            source_type='zip',
            total_pages=len(pages),
            metadata={
                'source_path': str(zip_path),
                'archive_name': zip_path.name
            }
        )
    
    def _load_from_pdf(self, pdf_path: Path) -> InputResult:
        """Load manga from a PDF file."""
        if not self._pdf_available:
            raise RuntimeError("PDF support not available. Install pdf2image.")
        
        logger.info(f"Loading manga from PDF: {pdf_path}")
        
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF pages to images
            pil_images = convert_from_path(pdf_path, dpi=150)
            
            pages = []
            for idx, pil_img in enumerate(pil_images):
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                image = np.array(pil_img)
                page = MangaPage(
                    index=idx,
                    image=image,
                    source_path=pdf_path,
                    metadata={'page_number': idx + 1}
                )
                pages.append(page)
                logger.debug(f"Loaded PDF page {idx}")
            
            logger.info(f"Loaded {len(pages)} pages from PDF")
            
            return InputResult(
                pages=pages,
                source_type='pdf',
                total_pages=len(pages),
                metadata={
                    'source_path': str(pdf_path),
                    'pdf_name': pdf_path.name
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise
