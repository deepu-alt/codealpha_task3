from deep_sort_realtime.deepsort_tracker import DeepSort


def initialize_tracker():

    tracker = DeepSort(max_age=30)

    return tracker


def update_tracker(tracker, detections, frame):

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    return tracks