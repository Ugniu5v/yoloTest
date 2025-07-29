from ultralytics import YOLO
import cv2
import easyocr

cap = cv2.VideoCapture("video10sec.mp4")
w_frame, h_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 3840×2160
x1, x2, y1, y2 = 600, 3072, 1200, 2160

model = YOLO("yolov8n.pt")  # for car detection
lp_model = YOLO("license_plate_detector.pt")  # for license plate
reader = easyocr.Reader(["en"])  # for license plate text

#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter("output.mp4", fourcc, fps, (int(w_frame), int(h_frame)))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        crop_frame = frame[y1:y2, x1:x2]
        results = model.track(crop_frame, conf=0.8, verbose=False, device="mps")[0]

        for box in results.boxes:
            car_crop_x1, car_crop_y1, car_crop_x2, car_crop_y2 = map(int, box.xyxy[0])
            car_global_x1 = car_crop_x1 + x1
            car_global_y1 = car_crop_y1 + y1
            car_global_x2 = car_crop_x2 + x1
            car_global_y2 = car_crop_y2 + y1

            label = f"Conf: {box.conf[0]:.2f} ID: {box.id[0]:.2f}"
            cv2.rectangle(frame, (car_global_x1, car_global_y1), (car_global_x2, car_global_y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (car_global_x1, car_global_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                1,
            )

            car_crop = frame[car_global_y1:car_global_y2, car_global_x1:car_global_x2]
            lp_result = lp_model.track(car_crop, verbose=False, device="mps")[0]

            if len(lp_result.boxes) > 0:
                lp_box = lp_result.boxes[0]
                lp_crop_x1, lp_crop_y1, lp_crop_x2, lp_crop_y2 = map(int, lp_box.xyxy[0])
                lp_global_x1 = lp_crop_x1 + car_global_x1
                lp_global_y1 = lp_crop_y1 + car_global_y1
                lp_global_x2 = lp_crop_x2 + car_global_x1
                lp_global_y2 = lp_crop_y2 + car_global_y1

                lp_crop = frame[lp_global_y1:lp_global_y2, lp_global_x1:lp_global_x2]
                ocr_result = reader.readtext(lp_crop)
                if ocr_result:
                    plate_text = (ocr_result[0][1]).upper()
                    plate_text = plate_text.replace(" ", "")
                    confidence = (ocr_result[0][2])
                    if confidence > 0.75 and len(plate_text) == 7:
                        print(f"license plate text: {plate_text}")
                else:
                    plate_text = "NA"
                    confidence = "NA"

                lp_label = f"Number plate: {plate_text}  confidence: {confidence: 0.2f}"
                cv2.rectangle(frame, (lp_global_x1, lp_global_y1), (lp_global_x2, lp_global_y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    lp_label,
                    (lp_global_x1, lp_global_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.5,
                    (255, 255, 255),
                    1,
                )

        #out.write(frame)
        cv2.imshow("Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
