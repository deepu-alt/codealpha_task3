def detect_objects(model, frame):

    results = model(frame)

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            # Format for Deep SORT
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                confidence,
                class_name
            ))

    return detections