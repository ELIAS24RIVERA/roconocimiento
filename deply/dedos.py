import cv2
import mediapipe as mp

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB (MediaPipe usa imágenes en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Realizar la detección de manos
    results = hands.process(frame_rgb)

    # Si se detectaron manos, dibujar las anotaciones
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de los dedos y las conexiones
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Aquí puedes obtener la posición de los dedos, por ejemplo, el índice
            # Los puntos de referencia de MediaPipe para los dedos están numerados
            # Según el índice del dedo, por ejemplo, el dedo índice es el 8 (segundo segmento del dedo índice)
            finger_tip = hand_landmarks.landmark[8]  # Punta del dedo índice (índice 8)
            h, w, c = frame.shape
            x, y = int(finger_tip.x * w), int(finger_tip.y * h)
            
            # Dibujar un círculo en la punta del dedo índice
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Puedes hacer lo mismo con otros dedos (índice, medio, anular, meñique, pulgar)

    # Mostrar el frame con las anotaciones de los dedos
    cv2.imshow("Detección de Dedos", frame)

    # Agregar un mecanismo para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Si presionas 'q', se cierra
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
