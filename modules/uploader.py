"""
Uploader Module

Handles uploading generated videos to various platforms.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

import cv2

from interfaces.base_uploader import (
    BaseUploader,
    Platform,
    UploadConfig,
    UploadResult
)

logger = logging.getLogger(__name__)


class Uploader(BaseUploader):
    """
    Concrete implementation of video uploader.
    
    Supports:
    - Local file copy
    - YouTube (via API or youtube-upload)
    - Placeholder for other platforms
    """
    
    PLATFORM_REQUIREMENTS = {
        Platform.YOUTUBE_SHORTS: {
            'max_duration': 60,
            'max_width': 1080,
            'max_height': 1920,
            'aspect_ratio': '9:16',
            'max_file_size_mb': 500
        },
        Platform.TIKTOK: {
            'max_duration': 60,
            'max_width': 1080,
            'max_height': 1920,
            'aspect_ratio': '9:16',
            'max_file_size_mb': 287
        },
        Platform.INSTAGRAM_REELS: {
            'max_duration': 90,
            'max_width': 1080,
            'max_height': 1920,
            'aspect_ratio': '9:16',
            'max_file_size_mb': 650
        },
        Platform.YOUTUBE: {
            'max_duration': 43200,  # 12 hours
            'max_width': 3840,
            'max_height': 2160,
            'max_file_size_mb': 256000  # 256 GB
        },
        Platform.TWITTER: {
            'max_duration': 140,
            'max_width': 1920,
            'max_height': 1200,
            'max_file_size_mb': 512
        },
        Platform.LOCAL: {
            'max_duration': float('inf'),
            'max_width': float('inf'),
            'max_height': float('inf'),
            'max_file_size_mb': float('inf')
        }
    }
    
    def __init__(self):
        self._credentials: Dict[Platform, Dict[str, str]] = {}
        self._authenticated: Dict[Platform, bool] = {}
    
    def get_supported_platforms(self) -> List[Platform]:
        """Get list of supported platforms."""
        return list(Platform)
    
    def get_platform_requirements(self, platform: Platform) -> Dict[str, any]:
        """Get requirements for a platform."""
        return self.PLATFORM_REQUIREMENTS.get(platform, {})
    
    def authenticate(self, platform: Platform, credentials: Dict[str, str]) -> bool:
        """Authenticate with a platform."""
        self._credentials[platform] = credentials
        
        if platform == Platform.LOCAL:
            self._authenticated[platform] = True
            return True
        
        if platform in [Platform.YOUTUBE, Platform.YOUTUBE_SHORTS]:
            return self._authenticate_youtube(credentials)
        
        # Placeholder authentication for other platforms
        logger.warning(f"Authentication for {platform.value} not implemented")
        self._authenticated[platform] = False
        return False
    
    def _authenticate_youtube(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with YouTube."""
        # Check for client secrets file
        client_secrets = credentials.get('client_secrets_file')
        
        if client_secrets and Path(client_secrets).exists():
            try:
                # Would use google-auth-oauthlib here
                logger.info("YouTube authentication would happen here")
                self._authenticated[Platform.YOUTUBE] = True
                self._authenticated[Platform.YOUTUBE_SHORTS] = True
                return True
            except Exception as e:
                logger.error(f"YouTube authentication failed: {e}")
        
        return False
    
    def is_authenticated(self, platform: Platform) -> bool:
        """Check if authenticated with a platform."""
        if platform == Platform.LOCAL:
            return True
        return self._authenticated.get(platform, False)
    
    def validate_video(self, video_path: Path, platform: Platform) -> Dict[str, any]:
        """Validate video meets platform requirements."""
        result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        requirements = self.get_platform_requirements(platform)
        
        if not video_path.exists():
            result['valid'] = False
            result['errors'].append(f"Video file not found: {video_path}")
            return result
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        max_size = requirements.get('max_file_size_mb', float('inf'))
        
        if file_size_mb > max_size:
            result['valid'] = False
            result['errors'].append(
                f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size}MB)"
            )
        
        # Check video properties
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                result['valid'] = False
                result['errors'].append("Could not open video file")
                return result
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps > 0 else 0
            
            cap.release()
            
            # Check duration
            max_duration = requirements.get('max_duration', float('inf'))
            if duration > max_duration:
                result['valid'] = False
                result['errors'].append(
                    f"Duration ({duration:.1f}s) exceeds limit ({max_duration}s)"
                )
            
            # Check dimensions
            max_width = requirements.get('max_width', float('inf'))
            max_height = requirements.get('max_height', float('inf'))
            
            if width > max_width or height > max_height:
                result['warnings'].append(
                    f"Video dimensions ({width}x{height}) may exceed recommended "
                    f"({max_width}x{max_height})"
                )
            
            # Store video info
            result['video_info'] = {
                'width': width,
                'height': height,
                'duration': duration,
                'fps': fps,
                'file_size_mb': file_size_mb
            }
            
        except Exception as e:
            result['warnings'].append(f"Could not validate video properties: {e}")
        
        return result
    
    def upload(
        self,
        video_path: Path,
        platform: Platform,
        config: UploadConfig
    ) -> UploadResult:
        """Upload video to specified platform."""
        logger.info(f"Uploading to {platform.value}: {video_path}")
        
        # Validate video
        validation = self.validate_video(video_path, platform)
        if not validation['valid']:
            return UploadResult(
                success=False,
                platform=platform,
                error_message="; ".join(validation['errors'])
            )
        
        # Handle different platforms
        if platform == Platform.LOCAL:
            return self._upload_local(video_path, config)
        
        if platform in [Platform.YOUTUBE, Platform.YOUTUBE_SHORTS]:
            return self._upload_youtube(video_path, config, platform)
        
        # Placeholder for other platforms
        return UploadResult(
            success=False,
            platform=platform,
            error_message=f"Upload to {platform.value} not implemented"
        )
    
    def _upload_local(self, video_path: Path, config: UploadConfig) -> UploadResult:
        """Copy video to local destination."""
        destination = config.metadata.get('destination')
        
        if not destination:
            # Use default outputs folder
            destination = Path('outputs') / f"{config.title.replace(' ', '_')}.mp4"
        
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(video_path, destination)
            
            return UploadResult(
                success=True,
                platform=Platform.LOCAL,
                video_url=str(destination.absolute()),
                upload_time=datetime.now().isoformat(),
                metadata={'destination': str(destination)}
            )
        except Exception as e:
            return UploadResult(
                success=False,
                platform=Platform.LOCAL,
                error_message=str(e)
            )
    
    def _upload_youtube(
        self, 
        video_path: Path, 
        config: UploadConfig,
        platform: Platform
    ) -> UploadResult:
        """Upload to YouTube."""
        if not self.is_authenticated(platform):
            return UploadResult(
                success=False,
                platform=platform,
                error_message="Not authenticated with YouTube"
            )
        
        # Check for youtube-upload CLI tool
        try:
            # Build command
            cmd = [
                'youtube-upload',
                '--title', config.title,
                '--description', config.description,
                '--privacy', config.visibility,
            ]
            
            if config.tags:
                cmd.extend(['--tags', ','.join(config.tags)])
            
            if config.category:
                cmd.extend(['--category', config.category])
            
            if config.thumbnail_path:
                cmd.extend(['--thumbnail', str(config.thumbnail_path)])
            
            cmd.append(str(video_path))
            
            # Execute upload
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                video_id = result.stdout.strip()
                return UploadResult(
                    success=True,
                    platform=platform,
                    video_id=video_id,
                    video_url=f"https://youtube.com/watch?v={video_id}",
                    upload_time=datetime.now().isoformat()
                )
            else:
                return UploadResult(
                    success=False,
                    platform=platform,
                    error_message=result.stderr
                )
                
        except FileNotFoundError:
            logger.warning("youtube-upload not found. Install with: pip install youtube-upload")
            return UploadResult(
                success=False,
                platform=platform,
                error_message="youtube-upload CLI not installed"
            )
        except Exception as e:
            return UploadResult(
                success=False,
                platform=platform,
                error_message=str(e)
            )
    
    def generate_thumbnail(
        self,
        video_path: Path,
        output_path: Path,
        timestamp: float = 0.0
    ) -> Path:
        """Generate thumbnail from video."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            # Seek to timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read frame")
            
            # Save thumbnail
            cv2.imwrite(str(output_path), frame)
            
            logger.info(f"Generated thumbnail: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            raise
