"""
Base Uploader Interface

Defines the contract for uploading generated videos to various platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from enum import Enum


class Platform(Enum):
    """Supported upload platforms."""
    YOUTUBE = "youtube"
    YOUTUBE_SHORTS = "youtube_shorts"
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    TWITTER = "twitter"
    LOCAL = "local"  # Just copy to a destination folder


@dataclass
class UploadConfig:
    """Configuration for video upload."""
    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    visibility: str = "private"  # private, unlisted, public
    category: str = ""
    thumbnail_path: Optional[Path] = None
    scheduled_time: Optional[str] = None  # ISO format datetime
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class UploadResult:
    """Result of video upload."""
    success: bool
    platform: Platform
    video_url: Optional[str] = None
    video_id: Optional[str] = None
    error_message: Optional[str] = None
    upload_time: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)


class BaseUploader(ABC):
    """
    Abstract base class for video uploaders.
    
    Implementations should handle uploading to various
    social media platforms.
    """
    
    @abstractmethod
    def upload(
        self,
        video_path: Path,
        platform: Platform,
        config: UploadConfig
    ) -> UploadResult:
        """
        Upload video to specified platform.
        
        Args:
            video_path: Path to video file
            platform: Target platform
            config: Upload configuration
            
        Returns:
            UploadResult with upload status
        """
        pass
    
    @abstractmethod
    def authenticate(self, platform: Platform, credentials: Dict[str, str]) -> bool:
        """
        Authenticate with a platform.
        
        Args:
            platform: Platform to authenticate with
            credentials: Platform-specific credentials
            
        Returns:
            True if authentication successful
        """
        pass
    
    @abstractmethod
    def is_authenticated(self, platform: Platform) -> bool:
        """
        Check if authenticated with a platform.
        
        Args:
            platform: Platform to check
            
        Returns:
            True if currently authenticated
        """
        pass
    
    @abstractmethod
    def validate_video(self, video_path: Path, platform: Platform) -> Dict[str, any]:
        """
        Validate video meets platform requirements.
        
        Args:
            video_path: Path to video file
            platform: Target platform
            
        Returns:
            Dict with validation results and any warnings
        """
        pass
    
    @abstractmethod
    def get_platform_requirements(self, platform: Platform) -> Dict[str, any]:
        """
        Get requirements for a platform.
        
        Args:
            platform: Platform to get requirements for
            
        Returns:
            Dict with platform requirements (dimensions, duration, etc.)
        """
        pass
    
    @abstractmethod
    def get_supported_platforms(self) -> List[Platform]:
        """
        Get list of supported platforms.
        
        Returns:
            List of supported platforms
        """
        pass
    
    @abstractmethod
    def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Path,
        timestamp: float = 0.0
    ) -> Path:
        """
        Generate thumbnail from video.
        
        Args:
            video_path: Path to video file
            output_path: Path to save thumbnail
            timestamp: Time in video to capture
            
        Returns:
            Path to generated thumbnail
        """
        pass
