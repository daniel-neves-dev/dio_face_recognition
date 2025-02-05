import os
import cv2
import numpy as np
from deepface import DeepFace

# =========================================
# CONFIGURAÇÕES
# =========================================
KNOWN_PEOPLE_DIR = "/home/daniel/PycharmProjects/dio_face/known_people"  # pasta com fotos das pessoas conhecidas
TEST_IMAGE_PATH = "mwk.jpg"  # imagem para teste
MODEL_NAME = "VGG-Face"  # modelo pré-treinado (ex: "VGG-Face", "Facenet", "ArcFace", etc.)
DISTANCE_THRESHOLD = 1.2 # limiar para decidir se é a pessoa ou não
DETECTOR_BACKEND = "mtcnn"  # detetor para extrair faces (ex: "mtcnn", "retinaface", etc.)

# =========================================
# PARTE 1: Carregar embeddings das pessoas conhecidas
# =========================================
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(KNOWN_PEOPLE_DIR):
    # Verifica se é arquivo de imagem
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path_img = os.path.join(KNOWN_PEOPLE_DIR, file_name)
    name = os.path.splitext(file_name)[0]  # Remove a extensão para obter o nome (rótulo)

    try:
        # Gera o embedding com DeepFace (assumindo 1 rosto principal por imagem)
        embedding_objs = DeepFace.represent(
            img_path=path_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True  # dispara erro se não achar rosto
        )

        if len(embedding_objs) > 0:
            # Convertendo a lista em array NumPy
            embedding_vector = np.array(embedding_objs[0]["embedding"])
            known_face_encodings.append(embedding_vector)
            known_face_names.append(name)
        else:
            print(f"[Aviso] Não foi possível extrair o embedding de: {file_name}")
    except Exception as e:
        print(f"[Erro] Ao processar {file_name}: {e}")

print("Pessoas conhecidas:", known_face_names)

# =========================================
# PARTE 2: Detectar rostos na imagem de teste e extrair seus embeddings
# =========================================
faces_data = DeepFace.extract_faces(
    img_path=TEST_IMAGE_PATH,
    detector_backend=DETECTOR_BACKEND,
    enforce_detection=False  # se False, não dá erro se não houver rosto
)

# Carrega a imagem para desenhar bounding boxes
image_bgr = cv2.imread(TEST_IMAGE_PATH)
if image_bgr is None:
    raise ValueError(f"Não foi possível abrir a imagem de teste: {TEST_IMAGE_PATH}")

# =========================================
# PARTE 3: Para cada rosto encontrado, comparar com as pessoas conhecidas
# =========================================
for face_info in faces_data:
    # face_info é um dicionário com:
    #  - "face": imagem recortada (array NumPy)
    #  - "facial_area": bounding box (x, y, w, h)
    #  - "confidence": se o detector retornar
    #  - etc.
    face_img = face_info["face"]
    box = face_info["facial_area"]  # dict com keys: x, y, w, h
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    # Extrair embedding desse rosto recortado
    face_embedding_list = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        enforce_detection=False
    )

    # Se não tiver embedding, pula
    if len(face_embedding_list) == 0:
        continue

    # Converte para array NumPy (evita erro na subtração)
    face_embedding = np.array(face_embedding_list[0]["embedding"])

    # =========================================
    # COMPARAR COM TODAS AS PESSOAS CONHECIDAS
    # =========================================
    best_match_name = "Desconhecido"
    best_match_distance = float("inf")

    # Percorre cada embedding conhecido
    for known_encoding, known_name in zip(known_face_encodings, known_face_names):
        dist = np.linalg.norm(face_embedding - known_encoding)  # Distância Euclidiana
        if dist < best_match_distance:
            best_match_distance = dist
            best_match_name = known_name

    # Decidir se ultrapassa o limiar
    if best_match_distance > DISTANCE_THRESHOLD:
        best_match_name = "Desconhecido"

    # =========================================
    # DESENHAR NA IMAGEM
    # =========================================
    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 230), 2)
    text = f"{best_match_name} ({best_match_distance:.2f})"
    cv2.putText(
        image_bgr,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 230),
        2
    )

# =========================================
# PARTE 4: Exibir o resultado
# =========================================
cv2.imshow("Resultado", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


