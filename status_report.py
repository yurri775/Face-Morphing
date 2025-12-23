#!/usr/bin/env python3
"""
Real Faces Dataset Morphing Pipeline - Status Report
"""

import os
import sys
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def generate_report():
    morph_dir = 'results/morph_all'
    images_dir = 'images'
    frames_dir = os.path.join(morph_dir, 'frames')
    
    print('=' * 70)
    print('REAL FACES MORPHING PIPELINE - STATUS REPORT')
    print('=' * 70)
    
    # Count images
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f'\n1. DATASET')
        print(f'   Images in folder: {len(image_files)}')
        print(f'   Source: benin007/human-faces-real-sketch-synthetic')
        print(f'   Type: Real Faces (500 PNG images)')
    else:
        print(f'\n1. DATASET')
        print(f'   Images folder NOT FOUND')
    
    # Count videos
    if os.path.exists(morph_dir):
        mp4_files = [f for f in os.listdir(morph_dir) if f.endswith('.mp4')]
        print(f'\n2. MORPHING OUTPUT')
        print(f'   Videos generated: {len(mp4_files)}/50')
        
        if len(mp4_files) > 0:
            # Get total size
            total_size = sum(os.path.getsize(os.path.join(morph_dir, f)) 
                           for f in mp4_files) / (1024*1024)
            print(f'   Total size: {total_size:.1f} MB')
            
            # Show progress
            progress = (len(mp4_files) * 100.0) / 50
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = '[' + '=' * filled + '-' * (bar_length - filled) + ']'
            print(f'   Progress: {bar} {progress:.0f}%')
        
        print(f'   Location: {morph_dir}/')
    else:
        print(f'\n2. MORPHING OUTPUT')
        print(f'   Output folder NOT FOUND')
    
    # Count extracted frames
    if os.path.exists(frames_dir):
        frame_dirs = [d for d in os.listdir(frames_dir) 
                      if os.path.isdir(os.path.join(frames_dir, d))]
        
        total_frames = 0
        for fdir in frame_dirs:
            fpath = os.path.join(frames_dir, fdir)
            frames = [f for f in os.listdir(fpath) if f.endswith('.png')]
            total_frames += len(frames)
        
        print(f'\n3. EXTRACTED FRAMES')
        print(f'   Frame sets: {len(frame_dirs)}')
        print(f'   Total frames: {total_frames}')
        print(f'   Location: {frames_dir}/')
        
        if len(frame_dirs) > 0:
            print(f'   Examples:')
            for fdir in sorted(frame_dirs)[:3]:
                fpath = os.path.join(frames_dir, fdir)
                frames = [f for f in os.listdir(fpath) if f.endswith('.png')]
                print(f'      - {fdir}: {len(frames)} frames')
    else:
        print(f'\n3. EXTRACTED FRAMES')
        print(f'   Frames folder NOT YET CREATED')
        print(f'   (Run extract_frames_real_faces.py after morphing completes)')
    
    print(f'\n' + '=' * 70)
    print('NEXT STEPS:')
    print('  1. Wait for morphing to complete (processing all 499 pairs)')
    print('  2. Run: python extract_frames_real_faces.py')
    print('  3. Check results/morph_all/frames/ for complete frame sequences')
    print('=' * 70)

if __name__ == '__main__':
    generate_report()
