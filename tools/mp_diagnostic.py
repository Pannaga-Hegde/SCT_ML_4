import cv2
import mediapipe as mp
import time

def run_diagnostic():
    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
        
    print("Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR: Failed to read frame.")
            break
            
        start_time = time.time()
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = hands.process(rgb_frame)
        latency_ms = (time.time() - start_time) * 1000.0
        
        h, w, c = frame.shape
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw bbox
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                cv2.rectangle(frame, (int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords))), (0, 255, 0), 2)
        
        # Draw status
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status = "Hand: YES" if hand_detected else "Hand: NO"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("MediaPipe Diagnostic", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    run_diagnostic()
