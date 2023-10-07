import cv2
import numpy as np
from pythonosc import udp_client

# Configura el cliente OSC para enviar mensajes a una dirección y puerto específicos
client = udp_client.SimpleUDPClient("192.168.15.13", 5000)
address1 = "/mi/direccion/1"
address2 = "/mi/direccion/2"

# Cargamos el modelo YOLO preentrenado
directorio = "/home/pi/deteccion/media/"
net = cv2.dnn.readNet(directorio + 'yolov3.weights', directorio + 'yolov3.cfg')
#net = cv2.dnn.readNet('/Users/josue/Downloads/yolov3.weights', '/Users/josue/Downloads/yolov3.cfg')

# Configuramos las clases que YOLO puede detectar (personas en este caso)
classes = ["person"]

# Inicializamos la cámara con resolución más baja
#cap = cv2.VideoCapture("/Users/josue/Downloads/test2.mp4")
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Ancho (Width)
cap.set(4, 380)  # Alto (Height)

# Procesa cada quinto frame
frame_counter = 0
frame_skip = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % frame_skip == 0:
        # Realizamos la detección de personas con YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        detections = net.forward(layer_names)

        # Iteramos sobre las detecciones
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:  # Clase 0 es "person"
                    center_x = int(obj[0] * frame.shape[1])
                    center_y = int(obj[1] * frame.shape[0])
                    width = int(obj[2] * frame.shape[1])
                    height = int(obj[3] * frame.shape[0])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Dibujamos un rectángulo alrededor de la persona detectada
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                    # Imprimimos la posición de la persona detectada
                    y_actual = list({y})
                    y_promedio = int(sum(y_actual) / len(y_actual))
                    x_actual = list({x})
                    x_promedio = int(sum(x_actual) / len(x_actual))
                    #print(y_promedio)
                    print(f'Persona encontrada en posición: X={x}, Y={y}')
                    client.send_message(address1, [x_promedio, y_promedio])
        # Mostramos el frame con las personas detectadas
        #cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
