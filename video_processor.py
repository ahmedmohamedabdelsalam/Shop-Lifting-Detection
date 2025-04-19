import cv2
import numpy as np
import os
from pathlib import Path
import torch
from torchvision import transforms
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import sys
from tqdm import tqdm
import logging
from colorama import init, Fore, Style
import time
import json
import csv
from datetime import datetime

# Initialize colorama for colored terminal output
init(autoreset=True)  # Auto-reset colors after each print

# Configure logging with a simpler format for terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format for terminal
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Disable matplotlib interactive mode to avoid issues in terminal
plt.ioff()

class VideoProcessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224), save_results: bool = True):
        """
        Initialize the video processor with target size.
        
        Args:
            target_size (Tuple[int, int]): Target size for frame standardization (height, width)
            save_results (bool): Whether to save results to files
        """
        self.target_size = target_size
        self.save_results = save_results
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Set style for better visualization - using a valid style
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_theme()  # Apply seaborn theme without specifying style
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = Path(f"results_{timestamp}")
        self.images_dir = self.base_output_dir / "images"
        self.data_dir = self.base_output_dir / "data"
        
        if self.save_results:
            try:
                self.base_output_dir.mkdir(exist_ok=True)
                self.images_dir.mkdir(exist_ok=True)
                self.data_dir.mkdir(exist_ok=True)
                logger.info(f"{Fore.GREEN}Output directories created: {self.base_output_dir}{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to create output directories: {e}{Style.RESET_ALL}")
                # Fallback to current directory
                self.base_output_dir = Path(".")
                self.images_dir = self.base_output_dir
                self.data_dir = self.base_output_dir
                logger.info(f"{Fore.YELLOW}Using current directory for output{Style.RESET_ALL}")
        
        # Initialize results storage
        self.results = []

    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], int, int, int, float]:
        """
        Load a video file and return its frames and metadata.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Tuple containing:
            - List of frames
            - Frame count
            - Frame width
            - Frame height
            - FPS
        """
        try:
            logger.info(f"{Fore.CYAN}Opening video: {video_path}{Style.RESET_ALL}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.warning(f"{Fore.YELLOW}Invalid FPS detected in {video_path}, using default value of 30{Style.RESET_ALL}")
                fps = 30.0
                
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.warning(f"{Fore.YELLOW}Could not determine total frame count, will count frames as they are read{Style.RESET_ALL}")
                total_frames = None
            
            # Use tqdm for progress bar
            pbar = tqdm(total=total_frames, desc=f"{Fore.CYAN}Loading frames{Style.RESET_ALL}", unit="frame", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            
            if not frames:
                raise ValueError("No frames found in video")
            
            logger.info(f"{Fore.GREEN}Successfully loaded {frame_count} frames{Style.RESET_ALL}")
            height, width = frames[0].shape[:2]
            return frames, frame_count, width, height, fps
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error loading video {video_path}: {str(e)}{Style.RESET_ALL}")
            raise

    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess frames by resizing and normalizing.
        
        Args:
            frames (List[np.ndarray]): List of video frames
            
        Returns:
            List of preprocessed frames
        """
        try:
            processed_frames = []
            pbar = tqdm(total=len(frames), desc=f"{Fore.CYAN}Preprocessing frames{Style.RESET_ALL}", unit="frame",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for frame in frames:
                # Resize frame
                resized = cv2.resize(frame, self.target_size)
                # Convert to RGB if needed
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                elif resized.shape[2] == 4:
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
                processed_frames.append(resized)
                pbar.update(1)
            
            pbar.close()
            logger.info(f"{Fore.GREEN}Successfully preprocessed {len(processed_frames)} frames{Style.RESET_ALL}")
            return processed_frames
        except Exception as e:
            logger.error(f"{Fore.RED}Error preprocessing frames: {str(e)}{Style.RESET_ALL}")
            raise

    def visualize_frames(self, frames: List[np.ndarray], video_info: dict, num_frames: int = 4):
        """
        Visualize a subset of frames from the video with enhanced information.
        
        Args:
            frames (List[np.ndarray]): List of video frames
            video_info (dict): Dictionary containing video metadata
            num_frames (int): Number of frames to visualize
        """
        try:
            if num_frames > len(frames):
                num_frames = len(frames)
            
            logger.info(f"{Fore.CYAN}Creating visualization for {video_info['filename']}{Style.RESET_ALL}")
            
            # Calculate grid layout
            grid_size = int(np.ceil(np.sqrt(num_frames)))
            fig = plt.figure(figsize=(15, 15))
            
            # Add title with video information
            title = f"Video Analysis: {video_info['filename']}\n"
            title += f"Duration: {timedelta(seconds=video_info['duration']):.2f} | "
            title += f"Frames: {video_info['frame_count']} | "
            title += f"Resolution: {video_info['width']}x{video_info['height']} | "
            title += f"FPS: {video_info['fps']:.2f}"
            fig.suptitle(title, fontsize=12, y=0.95)
            
            # Plot frames
            step = len(frames) // num_frames
            for i in range(num_frames):
                frame_idx = i * step
                ax = fig.add_subplot(grid_size, grid_size, i + 1)
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
                ax.imshow(frame_rgb)
                
                # Add frame information
                time_at_frame = frame_idx / video_info['fps']
                ax.set_title(f'Frame {frame_idx}\nTime: {timedelta(seconds=time_at_frame):.2f}')
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            if self.save_results:
                output_path = self.images_dir / f"{video_info['filename']}_analysis.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                logger.info(f"{Fore.GREEN}Visualization saved to {output_path}{Style.RESET_ALL}")
                
                # Save individual frames
                for i in range(num_frames):
                    frame_idx = i * step
                    frame_path = self.images_dir / f"{video_info['filename']}_frame_{frame_idx}.png"
                    cv2.imwrite(str(frame_path), frames[frame_idx])
                
                logger.info(f"{Fore.GREEN}Individual frames saved to {self.images_dir}{Style.RESET_ALL}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error visualizing frames: {str(e)}{Style.RESET_ALL}")
            # Try to save a simple version if the detailed one fails
            if self.save_results:
                try:
                    logger.info(f"{Fore.YELLOW}Attempting to save a simplified visualization{Style.RESET_ALL}")
                    if len(frames) > 0:
                        # Save just the first frame
                        first_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                        plt.figure(figsize=(10, 10))
                        plt.imshow(first_frame)
                        plt.title(f"First frame of {video_info['filename']}")
                        plt.axis('off')
                        simple_output_path = self.images_dir / f"{video_info['filename']}_simple.png"
                        plt.savefig(simple_output_path)
                        plt.close()
                        logger.info(f"{Fore.GREEN}Simplified visualization saved to {simple_output_path}{Style.RESET_ALL}")
                except Exception as e2:
                    logger.error(f"{Fore.RED}Failed to save simplified visualization: {str(e2)}{Style.RESET_ALL}")

    def analyze_video_stats(self, frames: List[np.ndarray]) -> dict:
        """
        Analyze basic statistics of the video frames.
        
        Args:
            frames (List[np.ndarray]): List of video frames
            
        Returns:
            Dictionary containing frame statistics
        """
        try:
            logger.info(f"{Fore.CYAN}Analyzing video statistics{Style.RESET_ALL}")
            # Convert frames to numpy array for faster computation
            frames_array = np.array(frames)
            
            stats = {
                'mean_brightness': float(np.mean(frames_array)),
                'std_brightness': float(np.std(frames_array)),
                'min_brightness': float(np.min(frames_array)),
                'max_brightness': float(np.max(frames_array)),
                'mean_motion': float(np.mean(np.abs(np.diff(frames_array, axis=0))))
            }
            logger.info(f"{Fore.GREEN}Statistics analysis complete{Style.RESET_ALL}")
            return stats
        except Exception as e:
            logger.error(f"{Fore.RED}Error analyzing video stats: {str(e)}{Style.RESET_ALL}")
            # Return default stats if analysis fails
            return {
                'mean_brightness': 0,
                'std_brightness': 0,
                'min_brightness': 0,
                'max_brightness': 0,
                'mean_motion': 0
            }
    
    def save_results_to_json(self, results: List[Dict[str, Any]]):
        """
        Save results to a JSON file.
        
        Args:
            results (List[Dict[str, Any]]): List of result dictionaries
        """
        if not self.save_results:
            return
            
        try:
            output_path = self.data_dir / "results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"{Fore.GREEN}Results saved to JSON: {output_path}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving results to JSON: {str(e)}{Style.RESET_ALL}")
    
    def save_results_to_csv(self, results: List[Dict[str, Any]]):
        """
        Save results to a CSV file.
        
        Args:
            results (List[Dict[str, Any]]): List of result dictionaries
        """
        if not self.save_results or not results:
            return
            
        try:
            output_path = self.data_dir / "results.csv"
            with open(output_path, 'w', newline='') as f:
                # Get all possible keys from all dictionaries
                fieldnames = set()
                for result in results:
                    fieldnames.update(result.keys())
                
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(results)
                
            logger.info(f"{Fore.GREEN}Results saved to CSV: {output_path}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving results to CSV: {str(e)}{Style.RESET_ALL}")
    
    def save_summary_report(self, results: List[Dict[str, Any]]):
        """
        Save a summary report of the processing results.
        
        Args:
            results (List[Dict[str, Any]]): List of result dictionaries
        """
        if not self.save_results:
            return
            
        try:
            output_path = self.data_dir / "summary_report.txt"
            
            with open(output_path, 'w') as f:
                f.write("VIDEO PROCESSING SUMMARY REPORT\n")
                f.write("==============================\n\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Videos Processed: {len(results)}\n\n")
                
                # Count by category
                categories = {}
                for result in results:
                    category = result.get('category', 'Unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                f.write("Videos by Category:\n")
                for category, count in categories.items():
                    f.write(f"  - {category}: {count}\n")
                f.write("\n")
                
                # Summary statistics
                if results:
                    f.write("Summary Statistics:\n")
                    f.write(f"  - Average Duration: {sum(r.get('duration', 0) for r in results) / len(results):.2f} seconds\n")
                    f.write(f"  - Average Frame Count: {sum(r.get('frame_count', 0) for r in results) / len(results):.0f}\n")
                    f.write(f"  - Average Resolution: {sum(r.get('width', 0) for r in results) / len(results):.0f}x{sum(r.get('height', 0) for r in results) / len(results):.0f}\n")
                    f.write(f"  - Average FPS: {sum(r.get('fps', 0) for r in results) / len(results):.2f}\n\n")
                
                # List all processed videos
                f.write("Processed Videos:\n")
                for i, result in enumerate(results, 1):
                    f.write(f"{i}. {result.get('filename', 'Unknown')} - {result.get('category', 'Unknown')}\n")
                    f.write(f"   Duration: {result.get('duration', 0):.2f}s, Frames: {result.get('frame_count', 0)}, Resolution: {result.get('width', 0)}x{result.get('height', 0)}\n")
            
            logger.info(f"{Fore.GREEN}Summary report saved to: {output_path}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving summary report: {str(e)}{Style.RESET_ALL}")

def process_dataset(base_path: str, processor: VideoProcessor):
    """
    Process all videos in the dataset with enhanced visualization.
    
    Args:
        base_path (str): Path to the dataset root directory
        processor (VideoProcessor): Video processor instance
    """
    try:
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base_path}")
        
        logger.info(f"{Fore.CYAN}{'='*50}")
        logger.info(f"Starting dataset processing: {base_path}")
        logger.info(f"{'='*50}{Style.RESET_ALL}")
        
        for category in ['shop lifters', 'non shop lifters']:
            category_path = base_path / category
            if not category_path.exists():
                logger.warning(f"{Fore.YELLOW}Category directory not found: {category_path}{Style.RESET_ALL}")
                continue
            
            logger.info(f"\n{Fore.CYAN}{'='*50}")
            logger.info(f"Processing {category.upper()}")
            logger.info(f"{'='*50}{Style.RESET_ALL}")
            
            # Find all video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov']:
                video_files.extend(list(category_path.glob(ext)))
            
            if not video_files:
                logger.warning(f"{Fore.YELLOW}No video files found in {category_path}{Style.RESET_ALL}")
                continue
            
            logger.info(f"{Fore.GREEN}Found {len(video_files)} video files in {category}{Style.RESET_ALL}")
            
            for video_file in video_files:
                try:
                    logger.info(f"\n{Fore.GREEN}{'='*30} Processing: {video_file.name} {'='*30}{Style.RESET_ALL}")
                    start_time = time.time()
                    
                    frames, frame_count, width, height, fps = processor.load_video(str(video_file))
                    processed_frames = processor.preprocess_frames(frames)
                    
                    # Calculate video duration
                    duration = frame_count / fps
                    
                    # Prepare video info for visualization
                    video_info = {
                        'filename': video_file.name,
                        'frame_count': frame_count,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'duration': duration
                    }
                    
                    # Analyze video statistics
                    stats = processor.analyze_video_stats(processed_frames)
                    
                    # Print detailed information
                    logger.info(f"\n{Fore.YELLOW}Video Statistics:{Style.RESET_ALL}")
                    logger.info(f"Duration: {timedelta(seconds=duration):.2f}")
                    logger.info(f"Frame Count: {frame_count}")
                    logger.info(f"Resolution: {width}x{height}")
                    logger.info(f"FPS: {fps:.2f}")
                    logger.info(f"\n{Fore.YELLOW}Frame Analysis:{Style.RESET_ALL}")
                    logger.info(f"Mean Brightness: {stats['mean_brightness']:.2f}")
                    logger.info(f"Brightness Std: {stats['std_brightness']:.2f}")
                    logger.info(f"Motion Level: {stats['mean_motion']:.2f}")
                    
                    # Visualize frames with enhanced information
                    processor.visualize_frames(processed_frames, video_info, num_frames=4)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    logger.info(f"{Fore.GREEN}Processing completed in {processing_time:.2f} seconds{Style.RESET_ALL}")
                    
                    # Store results
                    result = {
                        'filename': video_file.name,
                        'category': category,
                        'frame_count': frame_count,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'duration': duration,
                        'processing_time': processing_time,
                        'mean_brightness': stats['mean_brightness'],
                        'std_brightness': stats['std_brightness'],
                        'min_brightness': stats['min_brightness'],
                        'max_brightness': stats['max_brightness'],
                        'mean_motion': stats['mean_motion'],
                        'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    processor.results.append(result)
                    
                except Exception as e:
                    logger.error(f"{Fore.RED}Error processing {video_file.name}: {str(e)}{Style.RESET_ALL}")
                    logger.info(f"{Fore.YELLOW}Skipping to next video...{Style.RESET_ALL}")
                    
    except Exception as e:
        logger.error(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        raise
    
    # Save all results
    if processor.save_results and processor.results:
        processor.save_results_to_json(processor.results)
        processor.save_results_to_csv(processor.results)
        processor.save_summary_report(processor.results)
        logger.info(f"{Fore.GREEN}All results saved to {processor.base_output_dir}{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        logger.info(f"{Fore.CYAN}Initializing video processor...{Style.RESET_ALL}")
        # Initialize video processor with save_results=True to save all results
        processor = VideoProcessor(target_size=(224, 224), save_results=True)
        
        # Process the dataset
        dataset_path = "Shop DataSet"
        logger.info(f"{Fore.CYAN}Starting processing of dataset: {dataset_path}{Style.RESET_ALL}")
        process_dataset(dataset_path, processor)
        
        logger.info(f"{Fore.GREEN}{'='*50}")
        logger.info(f"Processing completed successfully!")
        logger.info(f"{'='*50}{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"{Fore.RED}Program terminated due to error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1) 