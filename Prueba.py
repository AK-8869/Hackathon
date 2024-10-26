import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import language_tool_python
import speech_recognition as sr
import time
from PIL import Image, ImageDraw, ImageFont

# Cargar el modelo de detección de letras
model = load_model('dos.h5')

# Inicializar LanguageTool para corrección en español mexicano
tool = language_tool_python.LanguageTool('es')

# Definir las letras detectadas
letras = ['a', 'b', 'c', 'd', 'e','f,', 'g', 'h', 'i', 'j','k','l','m','n','ñ','o','p','q','r','s','t','u','v','w','x','y','z']

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(1)

# Definir tamaños para la interfaz
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
MINI_FRAME_WIDTH = 320
MINI_FRAME_HEIGHT = 240

# Inicializar variables
oracion = ""
oraciones_mostradas = []  # Lista para almacenar oraciones previas
umbral_confianza = .999
delay_deteccion = 2  # Delay de 1 segundo entre detecciones
ultima_deteccion = time.time()  # Inicializar con el tiempo actual
ultima_letra = time.time()  # Tiempo de la última letra detectada o espacio agregado
modo_persona_sorda = True  # Inicialmente en modo persona sorda (detectar señas)

# Crear un reconocedor de voz para conversión de voz a texto
recognizer = sr.Recognizer()

print("Sistema iniciado. Esperando activación de detección.")

# Crear función para dibujar texto con Pillow
def dibujar_texto(imagen, texto, posicion, fuente):
    pil_imagen = Image.fromarray(imagen)
    draw = ImageDraw.Draw(pil_imagen)
    draw.text(posicion, texto, font=fuente, fill=(255, 255, 255))
    return np.array(pil_imagen)

# Cargar fuente para Pillow (asegúrate de que la ruta sea correcta)
fuente = ImageFont.truetype("arial.ttf", 32)  # Cambia la ruta si es necesario

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al acceder a la cámara.")
            break

        # Procesar la imagen de la cámara
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Realizar detección de señas si estamos en modo persona sorda
        if modo_persona_sorda:
            # Detección de señas y predicción
            if results.multi_hand_landmarks:
                # Delay para evitar detecciones continuas
                if time.time() - ultima_deteccion > delay_deteccion:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Extraer landmarks
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append([landmark.x, landmark.y, landmark.z])

                        flattened_landmarks = np.array(landmarks).flatten().reshape(1, -1)

                        # Hacer predicción
                        prediccion = model.predict(flattened_landmarks)
                        probabilidad_maxima = np.max(prediccion)
                        clase_predicha = np.argmax(prediccion, axis=1)

                        # Verificar umbral de confianza
                        if probabilidad_maxima > umbral_confianza:
                            letra_detectada = letras[clase_predicha[0]]
                            oracion += letra_detectada
                            ultima_deteccion = time.time()  # Actualizar el tiempo de la última detección
                            ultima_letra = time.time()  # Actualizar el tiempo de la última letra
                            print(f"Letra detectada: {letra_detectada}")
                            # Mostrar la letra en la pantalla usando Pillow
                            frame_bgr = dibujar_texto(frame_bgr, f"Letra: {letra_detectada}", (10, 30), fuente)
                        else:
                            frame_bgr = dibujar_texto(frame_bgr, "No se detecta letra clara", (10, 30), fuente)

            else:
                frame_bgr = dibujar_texto(frame_bgr, "No se detectan manos", (10, 30), fuente)

        # Redimensionar el frame para la mini captura de video
        mini_frame = cv2.resize(frame_bgr, (MINI_FRAME_WIDTH, MINI_FRAME_HEIGHT))

        # Crear la imagen de visualización
        display_image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Colocar la mini captura de video en la imagen de visualización
        display_image[10:10+MINI_FRAME_HEIGHT, 10:10+MINI_FRAME_WIDTH] = mini_frame

        # Mostrar la oración actual que se está formando
        display_image = dibujar_texto(display_image, f"Oración actual: {oracion}", (10, 10 + MINI_FRAME_HEIGHT + 30), fuente)
        
        # Mostrar todas las oraciones en la pantalla
        text_y = 10 + MINI_FRAME_HEIGHT + 30 + 30
        for i, oracion_prev in enumerate(oraciones_mostradas[::-1]):
            display_image = dibujar_texto(display_image, f"{oracion_prev}", (10, text_y), fuente)
            text_y += 30
            if text_y > DISPLAY_HEIGHT - 30:
                break  # Evitar escribir fuera del área de visualización

        # Mostrar la imagen de visualización
        cv2.imshow('Conversación en tiempo real', display_image)

        # Capturar teclas presionadas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saliendo...")
            break
        elif key == ord(' '):  # Tecla espaciadora presionada
            if modo_persona_sorda:
                # Enviar la oración de la persona sorda si está lista
                if oracion:
                    # Aplicar autocorrección
                    matches = tool.check(oracion)
                    oracion_corregida = language_tool_python.utils.correct(oracion, matches)
                    oraciones_mostradas.append("Sorda: " + oracion_corregida)
                    oracion = ""  # Reiniciar la oración actual
                    # Cambiar de modo
                    modo_persona_sorda = False
                    print("Turno cambiado a Persona Oyente.")
                else:
                    print("No hay oración para enviar.")
            else:
                # Modo persona oyente: grabar audio desde el micrófono de la PC
                print("Grabando audio...")
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    print("Di algo...")
                    audio = recognizer.listen(source)
                try:
                    texto = recognizer.recognize_google(audio, language="es-ES")
                    # Aplicar autocorrección
                    matches = tool.check(texto)
                    texto_corregido = language_tool_python.utils.correct(texto, matches)
                    print(f"Texto reconocido: {texto_corregido}")
                    oraciones_mostradas.append("Oyente: " + texto_corregido)
                    # Cambiar de modo
                    modo_persona_sorda = True
                    print("Turno cambiado a Persona Sorda.")
                except sr.UnknownValueError:
                    print("No se entendió el audio")
                except sr.RequestError as e:
                    print(f"Error en el servicio de reconocimiento de voz; {e}")
        elif key == 8:  # Tecla de borrar presionada
            oracion = ""  # Limpiar la oración actual
            print("Oración actual limpiada.")
        # Puedes agregar más condiciones para otras teclas si es necesario

    cap.release()
    cv2.destroyAllWindows()