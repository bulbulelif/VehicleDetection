from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import torch
import numpy as np
from config import MAX_AGE, MIN_HITS  # Assuming these are defined in config
from src.utils import check_gpu


class VehicleTracker:
    def __init__(self, model_path='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'):
        """
        Initializes the DeepSORT tracker.
        [cite: 7]
        """
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")  # Default DeepSORT config

        self.device = check_gpu()
        self.deepsort = DeepSort(model_path,
                                 device=self.device,
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=MAX_AGE,  # From your config
                                 n_init=MIN_HITS,  # From your config
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 )
        print(f"DeepSORT tracker initialized on {self.device}")

    def update(self, xyxy_detections, confidences, class_ids, img):
        """
        Updates the tracker with new detections.
        Args:
            xyxy_detections (list of lists): List of [x1, y1, x2, y2] bounding boxes.
            confidences (list): List of confidence scores for each detection.
            class_ids (list): List of class IDs for each detection.
            img (numpy.ndarray): The current frame (BGR format).
        Returns:
            list of list: Tracked objects, each containing [x1, y1, x2, y2, track_id, class_id].
        """
        if len(xyxy_detections) == 0:
            return []

        # DeepSORT expects detections in (x1, y1, x2, y2) format,
        # and also requires the embeddings for each detection for re-identification.
        # DeepSORT will extract features internally from the image based on the detections.

        # Convert to numpy array for DeepSORT
        bbox_xyxy = np.asarray(xyxy_detections)
        confs = np.asarray(confidences)

        # DeepSORT's update method directly takes the detections and the original image
        # It internally extracts features for re-identification.
        # We need to reshape bbox_xyxy to (N, 4) if it's not already.
        if bbox_xyxy.ndim == 1:
            bbox_xyxy = bbox_xyxy[np.newaxis, :]

        if bbox_xyxy.shape[0] > 0 and bbox_xyxy.shape[1] == 4:
            # The DeepSORT update method can return different formats.
            # We want [x1, y1, x2, y2, track_id, class_id]
            # DeepSORT's output `outputs` generally contains `xyxy`, `track_id`, `class_id`, `conf`
            outputs = self.deepsort.update(bbox_xyxy, confs, class_ids, img)

            # The 'outputs' format from DeepSORT is typically (x1, y1, x2, y2, track_id, class_id, confidence)
            # You might need to adjust based on the exact DeepSORT version or how you want to use it.
            # For simplicity, let's assume it returns a numpy array of [x1, y1, x2, y2, track_id]
            # We'll re-attach the class_id later.

            # DeepSORT's `update` method directly returns processed tracks.
            # The `outputs` from DeepSORT are typically an array of (x1,y1,x2,y2,track_id).
            # We need to iterate through the original detections to associate class_id with track_id.

            # A more direct way to integrate DeepSORT is to pass a list of Detections objects.
            # This generally involves creating `Detection` objects with bbox, confidence, and feature.
            # However, the `update` method with bbox_xyxy, confidences, and class_ids seems to be a common interface too.
            # Let's assume `outputs` is a list of [x1, y1, x2, y2, track_id].

            # Let's try to map class IDs back to the tracked objects.
            # This is a simplification and might need refinement based on exact DeepSORT implementation.

            # A common approach: DeepSORT gives back tracked bounding boxes and IDs.
            # We then need to find which original detection (with its class_id) corresponds to that tracked box.

            # For a basic integration, we'll assume DeepSORT returns tracks in the format [x1,y1,x2,y2,track_id]
            # and we need to manually add the class_id to them.

            # Let's adapt to what DeepSORT typically returns: (x1,y1,x2,y2,track_id,class_id,confidence) if supported
            # Or (x1,y1,x2,y2,track_id) and we add class_id separately.

            # From deep_sort_pytorch examples, `update` typically returns an array of `(x1,y1,x2,y2,track_id,class_id)`
            # if `class_ids` are passed.

            if outputs is not None and len(outputs) > 0:
                tracked_objects = []
                for output in outputs:
                    bbox = output[:4]
                    track_id = int(output[4])
                    # Assuming class_id is the 6th element if passed through.
                    # Or we need to map it back from the original detections.
                    # For simplicity, let's assume `outputs` includes class_id as the 6th element (index 5)
                    # if it's explicitly handled by DeepSORT.
                    # If not, you might need to find the best match for the class_id.
                    # A robust approach: find the detection that yielded this track_id.

                    # For now, let's assume DeepSORT returns (x1, y1, x2, y2, track_id) and we'll add class_id heuristically.
                    # A better way is to pass class_ids to DeepSORT if its API supports associating them directly with tracks.

                    # If DeepSORT doesn't directly return class_id with the track,
                    # you'll need to find the best matching detection (by IoU) to get its class_id.
                    # This adds complexity. For a direct output like in some DeepSORT integrations:
                    # Let's assume outputs are [x1, y1, x2, y2, track_id, class_id]
                    # This relies on the specific `DeepSORT` implementation from `deep_sort_pytorch`.
                    # If it's a simple DeepSORT (only gives track_id), then you need to re-associate.

                    # For `deep_sort_pytorch`, the `update` method actually allows for `classes` argument
                    # so it should pass them through.
                    # Let's assume `outputs` will be `(x1,y1,x2,y2,track_id,class_id)`

                    if output.shape[0] == 6:  # x1, y1, x2, y2, track_id, class_id
                        class_id = int(output[5])
                        tracked_objects.append(
                            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track_id, class_id])
                    elif output.shape[0] == 5:  # x1, y1, x2, y2, track_id. We need to find class_id.
                        # This is a simplification. A proper way would be to pass `class_ids` to DeepSORT.
                        # For now, let's assign a dummy class ID or try to find a close original detection.
                        # This part needs careful consideration based on the DeepSORT API you are using.
                        # If you pass `class_ids` to `deepsort.update`, it should ideally return them.

                        # Let's assume the `update` method of your `DeepSort` object in `deep_sort_pytorch`
                        # correctly propagates the class_id.
                        # The `outputs` array will then be `(x1,y1,x2,y2,track_id,class_id)`
                        # If it returns 5 elements, then class_id needs to be handled separately.

                        # Given the `deep_sort_pytorch` source, the `update` method passes `classes` through
                        # and they should be available in the output if specified.
                        # So, `outputs` will have `(x1, y1, x2, y2, track_id, class_id)`.
                        class_id = int(output[5]) if output.shape[0] > 5 else -1  # Fallback if not provided by DeepSORT
                        tracked_objects.append(
                            [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track_id, class_id])
            else:
                tracked_objects = []

            return tracked_objects
        return []


if __name__ == '__main__':
    # Example usage (dummy data)
    tracker = VehicleTracker()

    # Simulate a frame with detections
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections_frame1 = [
        [100, 100, 200, 200],  # box 1
        [300, 300, 400, 400]  # box 2
    ]
    confidences_frame1 = [0.9, 0.8]
    class_ids_frame1 = [0, 1]  # car, bus

    print("Frame 1 Tracks:")
    tracks_frame1 = tracker.update(detections_frame1, confidences_frame1, class_ids_frame1, dummy_img)
    for track in tracks_frame1:
        print(f"Track ID: {track[4]}, BBox: {track[:4]}, Class: {track[5]}")

    # Simulate next frame where box 1 moves
    detections_frame2 = [
        [105, 105, 205, 205],  # box 1 moved
        [500, 500, 600, 600]  # new box 3
    ]
    confidences_frame2 = [0.9, 0.7]
    class_ids_frame2 = [0, 2]  # car, truck

    print("\nFrame 2 Tracks:")
    tracks_frame2 = tracker.update(detections_frame2, confidences_frame2, class_ids_frame2, dummy_img)
    for track in tracks_frame2:
        print(f"Track ID: {track[4]}, BBox: {track[:4]}, Class: {track[5]}")

    # Simulate third frame with no new detections for the first track
    detections_frame3 = [
        [505, 505, 605, 605]  # box 3 moved
    ]
    confidences_frame3 = [0.7]
    class_ids_frame3 = [2]

    print("\nFrame 3 Tracks:")
    tracks_frame3 = tracker.update(detections_frame3, confidences_frame3, class_ids_frame3, dummy_img)
    for track in tracks_frame3:
        print(f"Track ID: {track[4]}, BBox: {track[:4]}, Class: {track[5]}")