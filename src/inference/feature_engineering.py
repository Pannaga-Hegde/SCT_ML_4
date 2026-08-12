"""Geometric Feature Engineering Engine for MediaPipe Hand Landmarks — GestureFlow.

Extracts 86 scale- and translation-invariant 3D geometric features from 21 hand landmarks:
- 63 normalized coordinates
- 5 fingertip-to-wrist distances
- 6 inter-fingertip pair distances (OK loop gap, C interior span)
- 1 fist compactness score
- 5 finger joint bend angles
- 3 palm normal orientation vector components
- 3 3D curvature and index extension metrics
"""

import numpy as np

def extract_rich_geometric_features(landmarks_21x3: np.ndarray) -> np.ndarray:
    """Extract 86 scale- and translation-invariant geometric features from 21x3 landmarks.
    
    Args:
        landmarks_21x3: Array of shape (21, 3) representing (x, y, z) keypoints.
        
    Returns:
        1D numpy array of 86 engineered features.
    """
    coords = landmarks_21x3.copy().astype(np.float32)
    
    # 1. Wrist translation centering
    wrist = coords[0].copy()
    coords -= wrist
    
    # 2. Scale normalization factor (Wrist L0 to Middle MCP L9 distance)
    middle_mcp = coords[9]
    scale_dist = float(np.linalg.norm(middle_mcp))
    if scale_dist < 1e-6:
        scale_dist = float(np.max(np.linalg.norm(coords, axis=1)))
        if scale_dist < 1e-6:
            scale_dist = 1.0
            
    coords /= scale_dist
    flat_63 = coords.flatten()
    
    # Indices of key landmarks
    # 0: Wrist
    # Thumb: 1, 2, 3, 4
    # Index: 5, 6, 7, 8
    # Middle: 9, 10, 11, 12
    # Ring: 13, 14, 15, 16
    # Pinky: 17, 18, 19, 20
    
    # 3. Fingertip to Wrist Distances (5 features)
    tips = [4, 8, 12, 16, 20]
    tip_dists = [float(np.linalg.norm(coords[t])) for t in tips]
    
    # 4. Inter-Fingertip Pair Distances (6 features)
    # 4-8: Thumb to Index (OK-loop / C-curve gap)
    # 4-12: Thumb to Middle
    # 4-16: Thumb to Ring
    # 4-20: Thumb to Pinky
    # 8-12: Index to Middle
    # 16-20: Ring to Pinky
    pair_dists = [
        float(np.linalg.norm(coords[4] - coords[8])),
        float(np.linalg.norm(coords[4] - coords[12])),
        float(np.linalg.norm(coords[4] - coords[16])),
        float(np.linalg.norm(coords[4] - coords[20])),
        float(np.linalg.norm(coords[8] - coords[12])),
        float(np.linalg.norm(coords[16] - coords[20])),
    ]
    
    # 5. Fist Compactness Score (1 feature)
    # Sum of fingertip distances to wrist (small for fist, large for open palm)
    compactness = float(sum(tip_dists))
    
    # 6. Finger Joint Bend Angles (5 features)
    def compute_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.arccos(cos_theta))
        
    finger_angles = [
        compute_angle(coords[2], coords[3], coords[4]),   # Thumb DIP
        compute_angle(coords[5], coords[6], coords[8]),   # Index PIP
        compute_angle(coords[9], coords[10], coords[12]), # Middle PIP
        compute_angle(coords[13], coords[14], coords[16]),# Ring PIP
        compute_angle(coords[17], coords[18], coords[20]) # Pinky PIP
    ]
    
    # 7. Palm Normal Orientation Vector (3 features)
    # Cross product of Wrist->IndexMCP (0->5) and Wrist->PinkyMCP (0->17)
    v_idx = coords[5]
    v_pnk = coords[17]
    palm_normal = np.cross(v_idx, v_pnk)
    norm_len = np.linalg.norm(palm_normal)
    if norm_len > 1e-6:
        palm_normal /= norm_len
    else:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    palm_norm_list = [float(palm_normal[0]), float(palm_normal[1]), float(palm_normal[2])]
    
    # 8. Curvature & Index Extension Metrics (3 features)
    z_span = float(coords[:, 2].max() - coords[:, 2].min())
    thumb_index_angle = compute_angle(coords[4], coords[0], coords[8])
    index_palm_angle = float(np.dot(coords[8], palm_normal))
    
    extra_metrics = [z_span, thumb_index_angle, index_palm_angle]
    
    # Combine all into single 86-element feature vector
    engineered_features = np.array(
        flat_63.tolist() + tip_dists + pair_dists + [compactness] + finger_angles + palm_norm_list + extra_metrics,
        dtype=np.float32
    )
    
    return engineered_features
