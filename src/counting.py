import cv2
import numpy as np
from collections import defaultdict
from src.utils import intersects
from config import COUNTING_LINE_COORDS


class VehicleCounter:
    def __init__(self, class_names):
        self.counted_ids = set()
        self.vehicle_counts = defaultdict(int)
        self.class_names = class_names
        self.prev_centroids = {}  # Stores previous centroid for each track ID
        print(f"Vehicle counter initialized for classes: {class_names}")

    def update(self, tracked_objects):
        """
        Updates vehicle counts based on tracked objects crossing the counting line.
        [cite: 2]
        Args:
            tracked_objects (list): List of tracked objects, each
                                    [x1, y1, x2, y2, track_id, class_id].
        Returns:
            dict: Current counts for each vehicle type.
        """
        current_frame_centroids = {}

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, class_id = obj

            # Calculate centroid of the bounding box
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            current_frame_centroids[track_id] = (centroid_x, centroid_y)

            if track_id not in self.counted_ids:
                if track_id in self.prev_centroids:
                    # Check if the track crossed the line from previous frame to current frame
                    prev_cx, prev_cy = self.prev_centroids[track_id]

                    # Define line points (P1, P2)
                    line_p1 = (COUNTING_LINE_COORDS[0], COUNTING_LINE_COORDS[1])
                    line_p2 = (COUNTING_LINE_COORDS[2], COUNTING_LINE_COORDS[3])

                    # Check if the line segment from prev_centroid to current_centroid intersects the counting line
                    # This is a simplified check. A proper line segment intersection algorithm is better.
                    # For a horizontal line (y1 == y2), check if prev_cy is on one side and current_cy on the other.
                    if line_p1[1] == line_p2[1]:  # Horizontal line
                        if (prev_cy < line_p1[1] and centroid_y >= line_p1[1]) or \
                                (prev_cy > line_p1[1] and centroid_y <= line_p1[1]):
                            # Check if the crossing happened within the X bounds of the line
                            if min(line_p1[0], line_p2[0]) <= centroid_x <= max(line_p1[0], line_p2[0]):
                                vehicle_type = self.class_names.get(class_id, f"Unknown_{class_id}")
                                self.vehicle_counts[vehicle_type] += 1
                                self.counted_ids.add(track_id)
                    # Add similar logic for vertical lines if needed
                    elif line_p1[0] == line_p2[0]:  # Vertical line
                        if (prev_cx < line_p1[0] and centroid_x >= line_p1[0]) or \
                                (prev_cx > line_p1[0] and centroid_x <= line_p1[0]):
                            if min(line_p1[1], line_p2[1]) <= centroid_y <= max(line_p1[1], line_p2[1]):
                                vehicle_type = self.class_names.get(class_id, f"Unknown_{class_id}")
                                self.vehicle_counts[vehicle_type] += 1
                                self.counted_ids.add(track_id)
                    else:  # Slanted line - more complex intersection logic needed for centroid path
                        # For simplicity, if not horizontal/vertical, use bounding box intersection for now.
                        # This is less precise for counting line crossings.
                        if intersects((x1, y1, x2, y2), COUNTING_LINE_COORDS):
                            vehicle_type = self.class_names.get(class_id, f"Unknown_{class_id}")
                            if track_id not in self.counted_ids:
                                self.vehicle_counts[vehicle_type] += 1
                                self.counted_ids.add(track_id)
                else:
                    # If this is the first time we see this track ID, just store its centroid
                    pass  # Don't count yet, wait for it to cross

        # Update previous centroids for the next frame
        self.prev_centroids = current_frame_centroids

        # Remove old IDs that are no longer tracked to prevent memory issues for long videos
        # (Optional, but good practice for real-time systems)
        current_track_ids = {obj[4] for obj in tracked_objects}
        self.counted_ids = self.counted_ids.intersection(current_track_ids)
        self.prev_centroids = {tid: self.prev_centroids[tid] for tid in current_track_ids if tid in self.prev_centroids}

        return self.vehicle_counts


if __name__ == '__main__':
    # Example Usage
    class_names_map = {0: 'car', 1: 'bus', 2: 'truck'}
    counter = VehicleCounter(class_names_map)

    # Simulate tracked objects over frames
    # Frame 1: Car appears above line
    tracked_objects_f1 = [
        [100, 200, 150, 250, 1, 0],  # car, track_id 1 (above line)
    ]
    counts_f1 = counter.update(tracked_objects_f1)
    print(f"Frame 1 Counts: {counts_f1}")  # Expected: {}

    # Frame 2: Car crosses line downwards
    tracked_objects_f2 = [
        [100, 310, 150, 360, 1, 0],  # car, track_id 1 (crossed line)
    ]
    counts_f2 = counter.update(tracked_objects_f2)
    print(f"Frame 2 Counts: {counts_f2}")  # Expected: {'car': 1}

    # Frame 3: Another vehicle appears
    tracked_objects_f3 = [
        [100, 320, 150, 370, 1, 0],  # car, track_id 1 (below line)
        [500, 200, 550, 250, 2, 1],  # bus, track_id 2 (above line)
    ]
    counts_f3 = counter.update(tracked_objects_f3)
    print(f"Frame 3 Counts: {counts_f3}")  # Expected: {'car': 1}

    # Frame 4: Bus crosses line
    tracked_objects_f4 = [
        [100, 330, 150, 380, 1, 0],  # car, track_id 1
        [500, 310, 550, 360, 2, 1],  # bus, track_id 2 (crossed line)
    ]
    counts_f4 = counter.update(tracked_objects_f4)
    print(f"Frame 4 Counts: {counts_f4}")  # Expected: {'car': 1, 'bus': 1}

    # Frame 5: Car that was already counted
    tracked_objects_f5 = [
        [100, 340, 150, 390, 1, 0],
        [500, 320, 550, 370, 2, 1],
    ]
    counts_f5 = counter.update(tracked_objects_f5)
    print(f"Frame 5 Counts: {counts_f5}")  # Expected: {'car': 1, 'bus': 1} (no new counts)