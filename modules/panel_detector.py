"""
Panel Detector Module

Detects and extracts individual panels from manga pages.
Uses a combination of contour detection and heuristics.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from interfaces.base_panel_detector import (
    BasePanelDetector, Panel, BoundingBox, PanelDetectionResult
)

logger = logging.getLogger(__name__)


class PanelDetector(BasePanelDetector):
    """
    Concrete implementation of panel detection.
    
    Uses OpenCV for contour-based panel detection with
    configurable parameters.
    """
    
    def __init__(self):
        # Default detection parameters
        self._min_panel_area_ratio = 0.01  # Minimum panel area as ratio of page
        self._max_panel_area_ratio = 0.95  # Maximum panel area as ratio of page
        self._min_aspect_ratio = 0.1
        self._max_aspect_ratio = 10.0
        self._border_threshold = 20  # Pixels from edge to consider as border
        self._canny_low = 50
        self._canny_high = 150
        self._dilation_kernel_size = 3
        self._dilation_iterations = 2
    
    def set_detection_params(self, **kwargs) -> None:
        """Set detection parameters."""
        if 'min_panel_area_ratio' in kwargs:
            self._min_panel_area_ratio = kwargs['min_panel_area_ratio']
        if 'max_panel_area_ratio' in kwargs:
            self._max_panel_area_ratio = kwargs['max_panel_area_ratio']
        if 'min_aspect_ratio' in kwargs:
            self._min_aspect_ratio = kwargs['min_aspect_ratio']
        if 'max_aspect_ratio' in kwargs:
            self._max_aspect_ratio = kwargs['max_aspect_ratio']
        if 'border_threshold' in kwargs:
            self._border_threshold = kwargs['border_threshold']
        if 'canny_low' in kwargs:
            self._canny_low = kwargs['canny_low']
        if 'canny_high' in kwargs:
            self._canny_high = kwargs['canny_high']
        
        logger.debug(f"Updated detection parameters: {kwargs}")
    
    def detect(self, page_image: np.ndarray, page_index: int) -> PanelDetectionResult:
        """Detect panels in a single manga page."""
        logger.debug(f"Detecting panels on page {page_index}")
        
        height, width = page_image.shape[:2]
        page_area = height * width
        
        # Preprocess image
        processed = self._preprocess(page_image)
        
        # Find contours
        contours = self._find_panel_contours(processed)
        
        # Filter and create panels
        panels = []
        panel_idx = 0
        
        for contour in contours:
            bbox = self._contour_to_bbox(contour)
            
            # Filter based on area
            area_ratio = bbox.area / page_area
            if area_ratio < self._min_panel_area_ratio or area_ratio > self._max_panel_area_ratio:
                continue
            
            # Filter based on aspect ratio
            aspect = bbox.width / max(bbox.height, 1)
            if aspect < self._min_aspect_ratio or aspect > self._max_aspect_ratio:
                continue
            
            # Extract panel image
            panel_image = self._extract_panel(page_image, bbox)
            
            # Calculate confidence based on contour properties
            confidence = self._calculate_confidence(contour, bbox, page_area)
            
            panel = Panel(
                index=panel_idx,
                page_index=page_index,
                image=panel_image,
                bounding_box=bbox,
                confidence=confidence,
                reading_order=panel_idx,
                metadata={'area_ratio': area_ratio, 'aspect_ratio': aspect}
            )
            panels.append(panel)
            panel_idx += 1
        
        # If no panels detected, treat entire page as single panel
        if not panels:
            logger.warning(f"No panels detected on page {page_index}, using full page")
            panels = [self._create_full_page_panel(page_image, page_index)]
        
        # Determine reading order
        panels = self.determine_reading_order(panels, reading_direction="rtl")
        
        # Calculate overall detection confidence
        avg_confidence = sum(p.confidence for p in panels) / len(panels)
        
        logger.info(f"Detected {len(panels)} panels on page {page_index}")
        
        return PanelDetectionResult(
            page_index=page_index,
            panels=panels,
            total_panels=len(panels),
            detection_confidence=avg_confidence
        )
    
    def detect_batch(self, pages: List[Tuple[np.ndarray, int]]) -> List[PanelDetectionResult]:
        """Detect panels in multiple pages."""
        results = []
        for page_image, page_index in pages:
            result = self.detect(page_image, page_index)
            results.append(result)
        return results
    
    def determine_reading_order(
        self, 
        panels: List[Panel], 
        reading_direction: str = "rtl"
    ) -> List[Panel]:
        """
        Determine reading order of panels.
        
        For manga (RTL): top-to-bottom, right-to-left
        For comics (LTR): top-to-bottom, left-to-right
        """
        if not panels:
            return panels
        
        # Get page dimensions from first panel
        page_height = max(p.bounding_box.y + p.bounding_box.height for p in panels)
        
        # Calculate row threshold (panels within this vertical distance are same row)
        avg_height = sum(p.bounding_box.height for p in panels) / len(panels)
        row_threshold = avg_height * 0.5
        
        # Sort panels into rows
        sorted_panels = sorted(panels, key=lambda p: p.bounding_box.y)
        
        rows = []
        current_row = [sorted_panels[0]]
        current_row_y = sorted_panels[0].bounding_box.y
        
        for panel in sorted_panels[1:]:
            if abs(panel.bounding_box.y - current_row_y) <= row_threshold:
                current_row.append(panel)
            else:
                rows.append(current_row)
                current_row = [panel]
                current_row_y = panel.bounding_box.y
        rows.append(current_row)
        
        # Sort each row by x position
        ordered_panels = []
        for row in rows:
            if reading_direction == "rtl":
                # Right to left for manga
                row.sort(key=lambda p: -p.bounding_box.x)
            else:
                # Left to right for comics
                row.sort(key=lambda p: p.bounding_box.x)
            ordered_panels.extend(row)
        
        # Update reading order
        for idx, panel in enumerate(ordered_panels):
            panel.reading_order = idx
        
        return ordered_panels
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for panel detection."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)
        
        # Dilate to connect nearby edges
        kernel = np.ones(
            (self._dilation_kernel_size, self._dilation_kernel_size), 
            np.uint8
        )
        dilated = cv2.dilate(edges, kernel, iterations=self._dilation_iterations)
        
        return dilated
    
    def _find_panel_contours(self, processed: np.ndarray) -> List[np.ndarray]:
        """Find contours that might be panels."""
        contours, hierarchy = cv2.findContours(
            processed, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and approximate contours
        panel_contours = []
        for contour in contours:
            # Approximate to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Panels typically have 4+ vertices (rectangular-ish)
            if len(approx) >= 4:
                panel_contours.append(approx)
        
        return panel_contours
    
    def _contour_to_bbox(self, contour: np.ndarray) -> BoundingBox:
        """Convert contour to bounding box."""
        x, y, w, h = cv2.boundingRect(contour)
        return BoundingBox(x=x, y=y, width=w, height=h)
    
    def _extract_panel(self, page_image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Extract panel image from page."""
        return page_image[
            bbox.y:bbox.y + bbox.height,
            bbox.x:bbox.x + bbox.width
        ].copy()
    
    def _calculate_confidence(
        self, 
        contour: np.ndarray, 
        bbox: BoundingBox, 
        page_area: int
    ) -> float:
        """Calculate detection confidence for a panel."""
        # Start with base confidence
        confidence = 0.7
        
        # Bonus for rectangular shape (4 vertices after approximation)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            confidence += 0.15
        
        # Bonus for reasonable aspect ratio (between 0.3 and 3.0)
        aspect = bbox.width / max(bbox.height, 1)
        if 0.3 <= aspect <= 3.0:
            confidence += 0.1
        
        # Penalize very small or very large panels
        area_ratio = bbox.area / page_area
        if area_ratio < 0.05 or area_ratio > 0.8:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_full_page_panel(self, page_image: np.ndarray, page_index: int) -> Panel:
        """Create a panel from the full page."""
        height, width = page_image.shape[:2]
        return Panel(
            index=0,
            page_index=page_index,
            image=page_image.copy(),
            bounding_box=BoundingBox(x=0, y=0, width=width, height=height),
            confidence=0.5,
            reading_order=0,
            metadata={'is_full_page': True}
        )
