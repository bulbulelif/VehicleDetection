import numpy as np
import cv2
import time
from collections import defaultdict

class VehicleTracker:
    """
    A simple, robust tracker for vehicles that doesn't rely on external models.
    This tracker uses a combination of IoU-based tracking and basic feature matching.
    """
    
    def __init__(self):
        """
        Initialize the vehicle tracker.
        """
        # Tracking parameters
        self.max_age = 30  # Maximum frames a track can be lost before removal
        self.min_hits = 3  # Minimum detection hits to confirm a track
        self.iou_threshold = 0.3  # Minimum IoU for matching
        
        # Storage for tracks
        self.tracks = []
        self.next_id = 1
        
        print("Simple IoU-based vehicle tracker initialized (no external model required)")
        
    def _calculate_iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            boxA (list): First box in format [x1, y1, x2, y2]
            boxB (list): Second box in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both bounding boxes
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the IoU
        iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
        
        return iou
    
    def _match_detections_to_tracks(self, detections, class_ids):
        """
        Match current detections to existing tracks based on IoU.
        
        Args:
            detections (list): List of current detections [x1, y1, x2, y2]
            class_ids (list): List of class IDs for each detection
            
        Returns:
            tuple: Matched track indices, matched detection indices, unmatched track indices, unmatched detection indices
        """
        if not self.tracks or not detections:
            return [], [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix between all tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                # Only match detections of the same class
                if track['class_id'] == class_ids[d]:
                    iou_matrix[t, d] = self._calculate_iou(track['bbox'], detection)
                else:
                    iou_matrix[t, d] = 0
        
        # Perform greedy matching
        matched_indices = []
        
        # Sort IoUs in descending order
        for _ in range(min(len(self.tracks), len(detections))):
            # Find highest IoU
            t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            iou = iou_matrix[t, d]
            
            # If IoU is high enough, it's a match
            if iou > self.iou_threshold:
                matched_indices.append((t, d))
                iou_matrix[t, :] = 0  # Remove this track from consideration
                iou_matrix[:, d] = 0  # Remove this detection from consideration
            else:
                break  # No more matches above threshold
        
        # Extract the track and detection indices
        matched_track_idx = [t for t, _ in matched_indices]
        matched_detection_idx = [d for _, d in matched_indices]
        
        # Get unmatched tracks and detections
        unmatched_track_idx = [t for t in range(len(self.tracks)) if t not in matched_track_idx]
        unmatched_detection_idx = [d for d in range(len(detections)) if d not in matched_detection_idx]
        
        return matched_track_idx, matched_detection_idx, unmatched_track_idx, unmatched_detection_idx
    
    def update(self, bboxes, confidences, class_ids, frame):
        """
        Update the tracker with new detections.
        
        Args:
            bboxes (list): List of bounding boxes in [x1, y1, x2, y2] format
            confidences (list): List of confidence scores
            class_ids (list): List of class IDs
            frame (numpy.ndarray): Current frame
            
        Returns:
            list: List of tracked objects [x1, y1, x2, y2, track_id, class_id]
        """
        # Return empty list if no detections
        if not bboxes:
            # Age all tracks
            for track in self.tracks:
                track['time_since_update'] += 1
            
            # Remove old tracks
            self.tracks = [track for track in self.tracks if track['time_since_update'] <= self.max_age]
            
            # Return active tracks
            return []
        
        # Match detections to existing tracks
        matched_track_idx, matched_detection_idx, unmatched_track_idx, unmatched_detection_idx = self._match_detections_to_tracks(bboxes, class_ids)
        
        # Update matched tracks
        for t, d in zip(matched_track_idx, matched_detection_idx):
            # Update bounding box
            self.tracks[t]['bbox'] = bboxes[d]
            self.tracks[t]['confidence'] = confidences[d]
            self.tracks[t]['class_id'] = class_ids[d]
            self.tracks[t]['time_since_update'] = 0
            self.tracks[t]['hits'] += 1
            self.tracks[t]['age'] += 1
        
        # Age unmatched tracks
        for t in unmatched_track_idx:
            self.tracks[t]['time_since_update'] += 1
            self.tracks[t]['age'] += 1
        
        # Create new tracks for unmatched detections
        for d in unmatched_detection_idx:
            self.tracks.append({
                'track_id': self.next_id,
                'bbox': bboxes[d],
                'confidence': confidences[d],
                'class_id': class_ids[d],
                'time_since_update': 0,
                'hits': 1,
                'age': 1
            })
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track['time_since_update'] <= self.max_age]
        
        # Get all confirmed tracks (sufficient hits and recently updated)
        confirmed_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] == 0:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['track_id']
                class_id = track['class_id']
                confirmed_tracks.append([int(x1), int(y1), int(x2), int(y2), int(track_id), int(class_id)])
        
        return confirmed_tracks


# For testing
if __name__ == "__main__":
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Print confirmation
    print("Tracker initialized successfully. Ready for vehicle tracking.")