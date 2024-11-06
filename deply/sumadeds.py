import cv2
import mediapipe as mp

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

def is_finger_raised(hand_landmarks, finger_tip_idx, finger_base_idx):
    """
    Función para verificar si un dedo está levantado. 
    Compara la posición de la punta del dedo con la base del dedo.
    """
    finger_tip = hand_landmarks.landmark[finger_tip_idx]
    finger_base = hand_landmarks.landmark[finger_base_idx]

    # Si la punta del dedo está por encima de la base (en el eje Y), el dedo está levantado
    if finger_tip.y < finger_base.y:
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB (MediaPipe usa imágenes en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Realizar la detección de manos
    results = hands.process(frame_rgb)

    # Inicializar el contador de dedos levantados
    fingers_raised = 0

    # Si se detectaron manos, dibujar las anotaciones
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de los dedos y las conexiones
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Contar los dedos levantados
            # Usamos la siguiente lógica:
            # - Dedos: [1: Pulgar, 2: Índice, 3: Medio, 4: Anular, 5: Meñique]
            # Para cada dedo, verificamos si está levantado.
            if is_finger_raised(hand_landmarks, 4, 2):  # Pulgar (Índices: 4 para punta, 2 para base)
                fingers_raised += 1
            if is_finger_raised(hand_landmarks, 8, 6):  # Índice (Índices: 8 para punta, 6 para base)
                fingers_raised += 1
            if is_finger_raised(hand_landmarks, 12, 10):  # Medio (Índices: 12 para punta, 10 para base)
                fingers_raised += 1
            if is_finger_raised(hand_landmarks, 16, 14):  # Anular (Índices: 16 para punta, 14 para base)
                fingers_raised += 1
            if is_finger_raised(hand_landmarks, 20, 18):  # Meñique (Índices: 20 para punta, 18 para base)
                fingers_raised += 1

            # Mostrar el número de dedos levantados en la pantalla
            cv2.putText(frame, f'Dedos levantados: {fingers_raised}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame con las anotaciones de los dedos
    cv2.imshow("Detección de Dedos", frame)

    # Agregar un mecanismo para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Si presionas 'q', se cierra
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
