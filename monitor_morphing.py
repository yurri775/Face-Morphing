import os
import sys
import time
from pathlib import Path

def monitor_progress():
    """Monitor progress of morphing batch"""
    morph_dir = 'results/morph_all'
    images_dir = 'images'
    
    # Get total expected pairs
    image_files = sorted([f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_pairs = len(image_files) - 1
    
    print('Real Faces Morphing Progress Monitor')
    print('=' * 60)
    
    while True:
        # Count generated videos
        mp4_files = [f for f in os.listdir(morph_dir) if f.endswith('.mp4')]
        video_count = len(mp4_files)
        
        # Count frames
        frames_dir = os.path.join(morph_dir, 'frames')
        frame_dirs = []
        if os.path.exists(frames_dir):
            frame_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
        
        # Calculate total frames
        total_frames = 0
        for fdir in frame_dirs:
            fpath = os.path.join(frames_dir, fdir)
            frames = [f for f in os.listdir(fpath) if f.endswith('.png')]
            total_frames += len(frames)
        
        # Display
        progress_pct = (video_count * 100.0) / 50  # 50 = limit in quick version
        
        print(f'\nTimestamp: {time.strftime("%H:%M:%S")}')
        print(f'Videos generated: {video_count}/50 ({progress_pct:.0f}%)')
        print(f'Frame directories: {len(frame_dirs)}')
        print(f'Total frames extracted: {total_frames}')
        
        if video_count == 50:
            print('\n[*] All 50 pairs completed!')
            print('You can now run: python extract_frames_real_faces.py')
            break
        
        # Wait and refresh
        time.sleep(10)

if __name__ == '__main__':
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print('\n\nMonitoring stopped.')
