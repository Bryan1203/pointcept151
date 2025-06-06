"""
Diagnostic script to investigate pose normalization inconsistencies
"""

import numpy as np
import glob
import os

def diagnose_pose_normalization(data_root="/home/bryan/pointcloud_data/airport_q2/", num_scans=10):
    """Diagnose why some scans have different ranges after pose normalization"""
    
    print("=== POSE NORMALIZATION DIAGNOSTIC ===\n")
    
    # Load poses
    pose_files = glob.glob(os.path.join(data_root, "*poses.npy"))
    poses = np.load(pose_files[0])
    
    # Create pose dictionary
    poses_dict = {}
    for pose in poses:
        poses_dict[pose['index']] = {'R': pose['R'], 't': pose['t']}
    
    print(f"Loaded {len(poses_dict)} poses")
    
    # Get scan files
    scan_files = glob.glob(os.path.join(data_root, "*.npy"))
    scan_files = [f for f in scan_files if not f.endswith('poses.npy')]
    scan_files = scan_files[:num_scans]  # Test first N scans
    
    print(f"Testing {len(scan_files)} scans\n")
    
    # Original normalization function
    def original_normalization(xyz, R, t):
        return np.dot(R.T, (xyz - t).T).T
    
    results = []
    
    for scan_file in scan_files:
        filename = os.path.basename(scan_file)
        try:
            scan_index = int(os.path.splitext(filename)[0])
            
            if scan_index not in poses_dict:
                print(f"❌ {filename}: No pose found")
                continue
            
            # Load scan
            scan_data = np.load(scan_file)
            xyz = scan_data['xyz'].astype(np.float32)
            
            # Get pose
            pose = poses_dict[scan_index]
            R = pose['R']
            t = pose['t']
            
            # Apply normalization
            normalized_xyz = original_normalization(xyz, R, t)
            
            # Calculate statistics
            original_center = xyz.mean(axis=0)
            original_range = np.abs(xyz).max()
            
            normalized_center = normalized_xyz.mean(axis=0)
            normalized_range = np.abs(normalized_xyz).max()
            normalized_std = normalized_xyz.std(axis=0)
            
            # Store results
            result = {
                'filename': filename,
                'scan_index': scan_index,
                'num_points': xyz.shape[0],
                'original_center': original_center,
                'original_range': original_range,
                'pose_t': t,
                'pose_R_det': np.linalg.det(R),
                'normalized_center': normalized_center,
                'normalized_range': normalized_range,
                'normalized_std': normalized_std,
                'xyz_min': normalized_xyz.min(axis=0),
                'xyz_max': normalized_xyz.max(axis=0)
            }
            results.append(result)
            
            print(f"✅ {filename}:")
            print(f"   Points: {xyz.shape[0]}")
            print(f"   Original center: [{original_center[0]:.1f}, {original_center[1]:.1f}, {original_center[2]:.1f}]")
            print(f"   Pose t:          [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]")
            print(f"   Norm center:     [{normalized_center[0]:.3f}, {normalized_center[1]:.3f}, {normalized_center[2]:.3f}]")
            print(f"   Norm range:      [{normalized_xyz.min(axis=0)[0]:.1f}, {normalized_xyz.max(axis=0)[0]:.1f}] x")
            print(f"                    [{normalized_xyz.min(axis=0)[1]:.1f}, {normalized_xyz.max(axis=0)[1]:.1f}] y")
            print(f"   Norm max abs:    {normalized_range:.1f}")
            print(f"   R determinant:   {np.linalg.det(R):.6f}")
            print()
            
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
            continue
    
    if len(results) == 0:
        print("No valid results to analyze")
        return
    
    print("=== ANALYSIS ===")
    
    # Analyze centers
    centers = np.array([r['normalized_center'] for r in results])
    center_magnitudes = np.linalg.norm(centers, axis=1)
    
    print(f"\n1. Normalized Centers Analysis:")
    print(f"   Mean center magnitude: {center_magnitudes.mean():.3f}")
    print(f"   Max center magnitude:  {center_magnitudes.max():.3f}")
    print(f"   Std center magnitude:  {center_magnitudes.std():.3f}")
    
    if center_magnitudes.max() > 5.0:
        print("   ⚠️  WARNING: Some scans are not well-centered!")
        # Find worst offenders
        worst_idx = np.argmax(center_magnitudes)
        worst_result = results[worst_idx]
        print(f"   Worst: {worst_result['filename']} - center magnitude {center_magnitudes[worst_idx]:.3f}")
    
    # Analyze ranges
    ranges = np.array([r['normalized_range'] for r in results])
    print(f"\n2. Normalized Range Analysis:")
    print(f"   Mean range: {ranges.mean():.1f}")
    print(f"   Min range:  {ranges.min():.1f}")
    print(f"   Max range:  {ranges.max():.1f}")
    print(f"   Std range:  {ranges.std():.1f}")
    
    if ranges.std() > 20.0:
        print("   ⚠️  WARNING: Large variation in ranges!")
        # Find outliers
        range_mean = ranges.mean()
        outliers = [r for r, res in zip(ranges, results) if abs(r - range_mean) > 2 * ranges.std()]
        for i, outlier_range in enumerate(outliers):
            outlier_idx = np.where(ranges == outlier_range)[0][0]
            print(f"   Outlier: {results[outlier_idx]['filename']} - range {outlier_range:.1f}")
    
    # Analyze pose translations
    translations = np.array([r['pose_t'] for r in results])
    translation_magnitudes = np.linalg.norm(translations, axis=1)
    
    print(f"\n3. Pose Translation Analysis:")
    print(f"   Mean translation magnitude: {translation_magnitudes.mean():.1f}")
    print(f"   Min translation magnitude:  {translation_magnitudes.min():.1f}")
    print(f"   Max translation magnitude:  {translation_magnitudes.max():.1f}")
    print(f"   Translation std:            {translations.std(axis=0)}")
    
    # Check for potential issues
    print(f"\n4. Potential Issues:")
    
    # Issue 1: Inconsistent sensor positions
    if translation_magnitudes.std() > 100:
        print("   ⚠️  Large variation in sensor positions (pose.t)")
        print("       This could indicate the sensor moved significantly between scans")
    
    # Issue 2: Point cloud extent vs pose
    for result in results:
        # Distance from original center to pose translation
        center_to_pose_dist = np.linalg.norm(result['original_center'] - result['pose_t'])
        if center_to_pose_dist > 50:
            print(f"   ⚠️  {result['filename']}: Large distance between point cloud center and pose")
            print(f"       Point cloud center: {result['original_center']}")
            print(f"       Pose translation:   {result['pose_t']}")
            print(f"       Distance: {center_to_pose_dist:.1f}")
    
    # Issue 3: Bad rotation matrices
    bad_rotations = [r for r in results if abs(r['pose_R_det'] - 1.0) > 0.001]
    if bad_rotations:
        print(f"   ⚠️  {len(bad_rotations)} scans have bad rotation matrices:")
        for r in bad_rotations:
            print(f"       {r['filename']}: det(R) = {r['pose_R_det']:.6f}")
    
    print(f"\n=== RECOMMENDATIONS ===")
    
    max_center_mag = center_magnitudes.max()
    max_range = ranges.max()
    
    if max_center_mag > 2.0:
        print("1. Consider subtracting the mean after pose normalization:")
        print("   normalized_xyz = normalized_xyz - normalized_xyz.mean(axis=0)")
    
    if max_range > 150:
        print("2. Large coordinate ranges detected. Consider:")
        print(f"   - Using grid_size >= {max_range/500:.3f}")
        print("   - Or scaling coordinates to a smaller range")
    
    if ranges.std() > 30:
        print("3. High variation in ranges suggests:")
        print("   - Different scanning distances or patterns")
        print("   - Potential pose calibration issues")
        print("   - Consider per-scan normalization")

if __name__ == "__main__":
    diagnose_pose_normalization(num_scans=15)  # Test 15 scans