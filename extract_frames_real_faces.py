import os
import subprocess
from pathlib import Path
import sys
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
morph_dir = 'results/morph_all'
frames_dir = os.path.join(morph_dir, 'frames')
os.makedirs(frames_dir, exist_ok=True)

# Lister les vidéos MP4
mp4_files = sorted([f for f in os.listdir(morph_dir) if f.endswith('.mp4')])

print(f'Found {len(mp4_files)} MP4 videos')
print(f'Extracting frames to: {frames_dir}/')
print()

success_count = 0
failed_count = 0
total_frames = 0

for idx, mp4_file in enumerate(mp4_files, 1):
    video_path = os.path.join(morph_dir, mp4_file)
    # Remove .mp4 extension for folder name
    pair_name = mp4_file.replace('.mp4', '')
    frame_folder = os.path.join(frames_dir, pair_name)
    
    # skip if already extracted
    if os.path.exists(frame_folder) and len(os.listdir(frame_folder)) > 0:
        existing = len([f for f in os.listdir(frame_folder) if f.endswith('.png')])
        print(f'[{idx:4d}/{len(mp4_files)}] SKIP {pair_name} ({existing} frames exist)')
        success_count += 1
        total_frames += existing
        continue
    
    os.makedirs(frame_folder, exist_ok=True)
    
    print(f'[{idx:4d}/{len(mp4_files)}] Extracting: {mp4_file}', end=' ... ')
    sys.stdout.flush()
    
    try:
        # Extract frames using ffmpeg
        frame_pattern = os.path.join(frame_folder, 'frame_%05d.png')
        cmd = ['ffmpeg', '-i', video_path, '-y', frame_pattern]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check if frames were created
        frame_count = len([f for f in os.listdir(frame_folder) if f.endswith('.png')])
        
        if frame_count > 0:
            success_count += 1
            total_frames += frame_count
            print(f'{frame_count} frames')
        else:
            failed_count += 1
            print(f'FAILED')
    
    except subprocess.TimeoutExpired:
        failed_count += 1
        print(f'TIMEOUT')
    except Exception as e:
        failed_count += 1
        print(f'ERROR: {str(e)[:100]}')

print()
print('=' * 60)
print('SUMMARY')
print(f'   Success: {success_count}/{len(mp4_files)}')
print(f'   Failed:  {failed_count}/{len(mp4_files)}')
print(f'   Total frames: {total_frames}')
print(f'   Output: {frames_dir}/')
print('=' * 60)
