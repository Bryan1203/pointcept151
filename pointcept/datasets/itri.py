"""
NPY Point Cloud dataset with pose normalization

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ItriDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
        use_pose_normalization=True,  # NEW: Enable pose normalization
    ):
        self.ignore_index = ignore_index
        self.use_pose_normalization = use_pose_normalization
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        
        # Initialize attributes manually to avoid parent's .pth logic
        self.data_root = data_root
        self.split = split
        from .transform import Compose
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            from .transform import TRANSFORMS
            self.test_voxelize = (
                TRANSFORMS.build(self.test_cfg.voxelize)
                if self.test_cfg.voxelize is not None
                else None
            )
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop is not None
                else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        # Use our custom data list method
        self.data_list = self.get_data_list()
        
        # NEW: Load pose data if using pose normalization
        if self.use_pose_normalization:
            self.poses_dict = self.load_poses()
        
        from pointcept.utils.logger import get_root_logger
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def load_poses(self):
        """Load all pose files and create a mapping from scan index to pose"""
        poses_dict = {}
        pose_files = glob.glob(os.path.join(self.data_root, "*poses.npy"))
        
        print(f"Loading pose files from: {self.data_root}")
        print(f"Found {len(pose_files)} pose files")
        
        for pose_file in pose_files:
            try:
                poses = np.load(pose_file)
                print(f"Loaded poses from {os.path.basename(pose_file)}: {len(poses)} poses")
                
                # Extract the sequence identifier from pose filename
                # Assuming pose files are named like "sequence_poses.npy" or just "poses.npy"
                pose_basename = os.path.basename(pose_file)
                
                for pose_entry in poses:
                    scan_index = pose_entry['index']
                    R = pose_entry['R']  # 3x3 rotation matrix
                    t = pose_entry['t']  # 3D translation vector
                    
                    # Store pose with scan index as key
                    poses_dict[scan_index] = {'R': R, 't': t}
                
            except Exception as e:
                print(f"Error loading pose file {pose_file}: {e}")
                continue
        
        print(f"Loaded poses for {len(poses_dict)} scans")
        return poses_dict

    def get_scan_index_from_filename(self, npy_file_path):
        """Extract scan index from NPY filename"""
        filename = os.path.basename(npy_file_path)
        # Assuming your NPY files are named like "10676.npy", "10532.npy", etc.
        scan_index = int(os.path.splitext(filename)[0])
        return scan_index

    def normalize_coordinates_with_pose(self, xyz, scan_index):
        """Normalize world coordinates to local coordinates using pose"""
        if not self.use_pose_normalization or scan_index not in self.poses_dict:
            print(f"Warning: No pose found for scan {scan_index}, using original coordinates")
            return xyz
        
        pose = self.poses_dict[scan_index]
        R = pose['R']  # 3x3 rotation matrix
        t = pose['t']  # 3D translation vector
        
        # Apply your normalization function: f(xyz, R, t) = R.T @ (xyz - t).T
        normalized_xyz = np.dot(R.T, (xyz - t).T).T
        
        return normalized_xyz.astype(np.float32)

    def get_data_list(self):
        # Get all NPY files
        data_files = glob.glob(os.path.join(self.data_root, "*.npy"))
        
        # Filter out pose files
        data_files = [f for f in data_files if not f.endswith('poses.npy')]
        
        # Debug: print what we found
        print(f"Looking for NPY files in: {self.data_root}")
        print(f"Found {len(data_files)} NPY files (excluding pose files)")
        if len(data_files) == 0:
            print("No NPY files found! Check your data_root path.")
            return []
        
        # Sort by numeric filename
        try:
            data_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        except ValueError as e:
            print(f"Warning: Could not sort files numerically: {e}")
            data_files.sort()
        
        print(f"First few files: {[os.path.basename(f) for f in data_files[:5]]}")
        
        # Split into train (70%), val (15%), test (15%)
        num_files = len(data_files)
        num_train = int(num_files * 0.7)
        num_val = int(num_files * 0.15)
        
        # Shuffle for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(num_files)
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]
        
        if self.split == "train":
            selected_indices = train_indices
        elif self.split == "val":
            selected_indices = val_indices
        elif self.split == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        selected_files = [data_files[i] for i in selected_indices]
        print(f"Selected {len(selected_files)} files for {self.split} split")
        
        return selected_files

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        filename = os.path.basename(data_path)
        
        print(f"[{filename}] Loading NPY file...")
        
        try:
            # Load your structured NPY data
            scan_data = np.load(data_path)
            
            # Extract coordinates, intensity, and type labels
            coord = scan_data['xyz'].astype(np.float32)
            strength = scan_data['intensity'].reshape(-1, 1).astype(np.float32)
            segment = scan_data['label']['type'].astype(np.int32)
            
            # print(f"[{filename}] Loaded {coord.shape[0]} points")
            # print(f"[{filename}] Original coord range: X[{coord[:, 0].min():.3f}, {coord[:, 0].max():.3f}], "
            #       f"Y[{coord[:, 1].min():.3f}, {coord[:, 1].max():.3f}], "
            #       f"Z[{coord[:, 2].min():.3f}, {coord[:, 2].max():.3f}]")
            
            # NEW: Apply pose normalization
            if self.use_pose_normalization:
                scan_index = self.get_scan_index_from_filename(data_path)
                coord = self.normalize_coordinates_with_pose(coord, scan_index)
                
                # print(f"[{filename}] After pose normalization: X[{coord[:, 0].min():.3f}, {coord[:, 0].max():.3f}], "
                #       f"Y[{coord[:, 1].min():.3f}, {coord[:, 1].max():.3f}], "
                #       f"Z[{coord[:, 2].min():.3f}, {coord[:, 2].max():.3f}]")
            
            # Check for empty point clouds
            if coord.shape[0] == 0:
                print(f"[{filename}] ERROR: Empty point cloud!")
                coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                strength = np.array([[1.0]], dtype=np.float32)
                segment = np.array([0], dtype=np.int32)
            
            # Check for problematic coordinates (NaN, inf)
            if np.any(~np.isfinite(coord)):
                print(f"[{filename}] ERROR: Non-finite coordinates detected!")
                coord = np.nan_to_num(coord, nan=0.0, posinf=50.0, neginf=-50.0)
                print(f"[{filename}] Fixed non-finite coordinates")
            
            # Handle unmapped labels
            unique_labels = np.unique(segment)
            unmapped_labels = [label for label in unique_labels if label not in self.learning_map]
            if unmapped_labels:
                print(f"[{filename}] Found unmapped labels {unmapped_labels}")
                for label in unmapped_labels:
                    self.learning_map[label] = self.ignore_index
            
            # Apply learning map to labels
            segment_mapped = np.vectorize(self.learning_map.__getitem__)(segment).astype(np.int32)
            
            # Check for valid labels
            valid_mask = segment_mapped != self.ignore_index
            num_valid = valid_mask.sum()
            
            print(f"[{filename}] Valid points: {num_valid}/{len(segment_mapped)}")
            
            # If no valid labels, create at least one valid point
            if num_valid == 0:
                print(f"[{filename}] WARNING: No valid labels, adding dummy valid point")
                segment_mapped = np.array([0], dtype=np.int32)
                coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                strength = np.array([[1.0]], dtype=np.float32)
            
            # Validate label range for 12 classes
            min_label, max_label = segment_mapped.min(), segment_mapped.max()
            if min_label < -1 or max_label > 11:
                print(f"[{filename}] ERROR: Invalid label range [{min_label}, {max_label}]")
                segment_mapped = np.clip(segment_mapped, -1, 11)
            
            # Create data dict
            data_dict = dict(
                coord=coord, 
                strength=strength,
                segment=segment_mapped.reshape([-1])
            )
            
            # print(f"[{filename}] Success: Created data_dict")
            return data_dict
            
        except Exception as e:
            print(f"[{filename}] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal dummy sample to prevent crashes
            dummy_coord = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            dummy_strength = np.array([[1.0]], dtype=np.float32)
            dummy_segment = np.array([0], dtype=np.int32)
            print(f"[{filename}] Returning dummy sample to prevent crash")
            return dict(coord=dummy_coord, strength=dummy_strength, segment=dummy_segment)

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(file_path))[0]

    def __len__(self):
        return len(self.data_list) * self.loop

    @staticmethod
    def get_learning_map(ignore_index):
        # Map your type labels to all 12 training classes (0-11)
        learning_map = {
            -1: 0,  # unlabeled -> ignore
            1: 0,   # 'none' -> class 0
            2: 1,   # 'solid' -> class 1
            3: 2,   # 'broken' -> class 2
            4: 3,   # 'solid solid' -> class 3
            5: 4,   # 'solid broken' -> class 4
            6: 5,   # 'broken solid' -> class 5
            7: 6,   # 'broken broken' -> class 6
            8: 7,   # 'botts dots' -> class 7
            9: 8,   # 'grass' -> class 8
            10: 9,  # 'curb' -> class 9
            11: 10, # 'custom' -> class 10
            12: 11, # 'edge' -> class 11
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: -1,  # ignore -> unlabeled
            0: 1,   # class 0 -> 'none'
            1: 2,   # class 1 -> 'solid'
            2: 3,   # class 2 -> 'broken'
            3: 4,   # class 3 -> 'solid solid'
            4: 5,   # class 4 -> 'solid broken'
            5: 6,   # class 5 -> 'broken solid'
            6: 7,   # class 6 -> 'broken broken'
            7: 8,   # class 7 -> 'botts dots'
            8: 9,   # class 8 -> 'grass'
            9: 10,  # class 9 -> 'curb'
            10: 11, # class 10 -> 'custom'
            11: 12, # class 11 -> 'edge'
        }
        return learning_map_inv