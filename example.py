import cv2
import demo as eng_det

# Setup camera
cap = cv2.VideoCapture(0)
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret: break

    # Maintaining a list of frames
    frame_buffer.append(frame)
    if len(frame_buffer) > 75:
        frame_buffer.pop(0)

    # Thresholds
    thresholds = {
        "Boredom": 0.32,
        "Engagement": 0.65,
        "Confusion": 0.21,
        "Frustration": 0.17
    }

    # Pass in a list of frames and (optional) thresholds and receive 
    # an engagement label
    label = eng_det.get_engagement_label(frame_buffer, thresholds)

    # Display result
    cv2.putText(frame, f"Status: {label}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Demo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()