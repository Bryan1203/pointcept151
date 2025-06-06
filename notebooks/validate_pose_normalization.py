"""
Validation script for pose normalization
Run this independently to test your pose data and normalization
"""

import numpy as np
import glob
import os

def validate_pose_normalization(data_root="/home/bryan/pointcloud_data/airport_q2/"):
    """Validate that pose normalization is working correctly"""
    
    print("=== POSE NORMALIZATION VALIDATION ===\n")
    
    # 1. Check pose files
    pose_files = glob.glob(os.path.join(data_root, "*poses.npy"))
    print(f"1. Found {len(pose_files)} pose files:")
    for pf in pose_files:
        print(f"   - {os.path.basename(pf)}")
    
    if len(pose_files) == 0:
        print("ERROR: No pose files found!")
        return False
    
    # 2. Load and examine pose data structure
    pose_file = pose_files[0]  # Use first pose file
    print(f"\n2. Loading pose file: {os.path.basename(pose_file)}")
    
    try:
        poses = np.load(pose_file)
        print(f"   Poses loaded: {len(poses)} entries")
        print(f"   Data type: {poses.dtype}")
        print(f"   Fields: {poses.dtype.names}")
        
        # Examine first few poses
        print("\n   First 3 poses:")
        for i in range(min(3, len(poses))):
            pose = poses[i]
            print(f"   Pose {i}:")
            print(f"     Index: {pose['index']}")
            print(f"     R shape: {pose['R'].shape}")
            print(f"     t shape: {pose['t'].shape}")
            print(f"     R:\n{pose['R']}")
            print(f"     t: {pose['t']}")
            
            # Validate rotation matrix
            R = pose['R']
            det = np.linalg.det(R)
            is_orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            print(f"     R determinant: {det:.6f} (should be ~1.0)")
            print(f"     R is orthogonal: {is_orthogonal}")
            print()
        
    except Exception as e:
        print(f"ERROR loading pose file: {e}")
        return False
    
    # 3. Find a corresponding scan file
    scan_files = glob.glob(os.path.join(data_root, "*.npy"))
    scan_files = [f for f in scan_files if not f.endswith('poses.npy')]
    
    if len(scan_files) == 0:
        print("ERROR: No scan files found!")
        return False
    
    print(f"3. Found {len(scan_files)} scan files")
    
    # Find a scan file that has a corresponding pose
    test_scan = None
    test_pose = None
    
    for scan_file in scan_files[:10]:  # Check first 10 scan files
        filename = os.path.basename(scan_file)
        try:
            scan_index = int(os.path.splitext(filename)[0])
            
            # Find corresponding pose
            for pose in poses:
                if pose['index'] == scan_index:
                    test_scan = scan_file
                    test_pose = pose
                    break
            
            if test_scan is not None:
                break
                
        except ValueError:
            continue
    
    if test_scan is None:
        print("ERROR: No scan file with corresponding pose found!")
        print("Available scan indices:", [int(os.path.splitext(os.path.basename(f))[0]) for f in scan_files[:5]])
        print("Available pose indices:", poses['index'][:10])
        return False
    
    print(f"\n4. Testing with scan: {os.path.basename(test_scan)}")
    print(f"   Corresponding pose index: {test_pose['index']}")
    
    # 4. Load scan data and test normalization
    try:
        scan_data = np.load(test_scan)
        xyz = scan_data['xyz'].astype(np.float32)
        print(f"   Loaded {xyz.shape[0]} points")
        print(f"   XYZ shape: {xyz.shape}")
        
        # Original coordinate statistics
        print(f"\n5. Original coordinates:")
        print(f"   X range: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"   Y range: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"   Z range: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
        print(f"   Center: [{xyz[:, 0].mean():.3f}, {xyz[:, 1].mean():.3f}, {xyz[:, 2].mean():.3f}]")
        print(f"   Max abs: {np.abs(xyz).max():.3f}")
        
    except Exception as e:
        print(f"ERROR loading scan file: {e}")
        return False
    
    # 5. Apply pose normalization (your original function)
    def original_normalization(xyz, R, t):
        return np.dot(R.T, (xyz - t).T).T
    
    print(f"\n6. Applying pose normalization...")
    R = test_pose['R']
    t = test_pose['t']
    
    try:
        normalized_xyz = original_normalization(xyz, R, t)
        
        print(f"   Normalized coordinates:")
        print(f"   X range: [{normalized_xyz[:, 0].min():.3f}, {normalized_xyz[:, 0].max():.3f}]")
        print(f"   Y range: [{normalized_xyz[:, 1].min():.3f}, {normalized_xyz[:, 1].max():.3f}]")
        print(f"   Z range: [{normalized_xyz[:, 2].min():.3f}, {normalized_xyz[:, 2].max():.3f}]")
        print(f"   Center: [{normalized_xyz[:, 0].mean():.3f}, {normalized_xyz[:, 1].mean():.3f}, {normalized_xyz[:, 2].mean():.3f}]")
        print(f"   Max abs: {np.abs(normalized_xyz).max():.3f}")
        
        # Check if normalization makes sense
        original_range = np.abs(xyz).max()
        normalized_range = np.abs(normalized_xyz).max()
        
        print(f"\n7. Normalization analysis:")
        print(f"   Original max range: {original_range:.3f}")
        print(f"   Normalized max range: {normalized_range:.3f}")
        print(f"   Reduction factor: {original_range/normalized_range:.2f}x")
        
        # Test if the transformation is reasonable
        if normalized_range > 1000:
            print("   WARNING: Normalized coordinates still very large!")
        elif normalized_range > 100:
            print("   WARNING: Normalized coordinates quite large for typical grid sampling")
        elif normalized_range < 1:
            print("   WARNING: Normalized coordinates very small")
        else:
            print("   OK: Normalized coordinates in reasonable range")
        
        # 6. Test improved normalization
        print(f"\n8. Testing improved normalization (with scaling)...")
        
        # Center around origin
        coord_center = normalized_xyz.mean(axis=0)
        centered_xyz = normalized_xyz - coord_center
        
        # Scale to reasonable range
        coord_scale = np.abs(centered_xyz).max()
        if coord_scale > 0:
            scaled_xyz = centered_xyz * (25.0 / coord_scale)
        else:
            scaled_xyz = centered_xyz
        
        print(f"   After centering and scaling:")
        print(f"   X range: [{scaled_xyz[:, 0].min():.3f}, {scaled_xyz[:, 0].max():.3f}]")
        print(f"   Y range: [{scaled_xyz[:, 1].min():.3f}, {scaled_xyz[:, 1].max():.3f}]")
        print(f"   Z range: [{scaled_xyz[:, 2].min():.3f}, {scaled_xyz[:, 2].max():.3f}]")
        print(f"   Max abs: {np.abs(scaled_xyz).max():.3f}")
        
        # 7. Test grid coordinate computation
        print(f"\n9. Testing grid coordinate computation:")
        grid_sizes = [0.05, 0.1, 0.2, 0.4]
        
        for grid_size in grid_sizes:
            # For original normalized coords
            grid_coords_orig = (normalized_xyz / grid_size).astype(np.int32)
            max_grid_orig = np.abs(grid_coords_orig).max()
            
            # For scaled coords
            grid_coords_scaled = (scaled_xyz / grid_size).astype(np.int32)
            max_grid_scaled = np.abs(grid_coords_scaled).max()
            
            print(f"   Grid size {grid_size}:")
            print(f"     Original norm: max grid coord = {max_grid_orig}")
            print(f"     Scaled norm:   max grid coord = {max_grid_scaled}")
            
            if max_grid_orig > 2000:
                print(f"     WARNING: Original normalization gives very large grid coords!")
            if max_grid_scaled > 2000:
                print(f"     WARNING: Scaled normalization gives very large grid coords!")
        
        print(f"\n=== VALIDATION COMPLETE ===")
        print(f"Recommendation: Use grid_size >= {normalized_range/500:.3f} for original normalization")
        print(f"Or use scaled normalization with grid_size >= 0.05")
        
        return True
        
    except Exception as e:
        print(f"ERROR during normalization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run validation
    success = validate_pose_normalization()
    
    if success:
        print("\n✅ Pose normalization validation completed successfully")
    else:
        print("\n❌ Pose normalization validation failed")