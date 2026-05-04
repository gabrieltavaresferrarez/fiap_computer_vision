import cv2
from face_detection import *
from face_recogtion import *
from face_utils import *



# ------------------------
# CONSTANTES
# ------------------------
PROCESSING_RESIZE_FACTOR = 0.33


# ------------------------
# Estado do sistema
# ------------------------
i_name = 0
list_names = ['Gabriel', 'Pedro', 'Joao', 'Lucas', 'Jose', 'Vitoria']
state = {
    "step": 0,
    "mode": "identificar",  # ou "cadastrar"
    "face_path": 'data/face.jpeg', # imagem da face 
    "name": list_names[i_name],
    'eye_state' : 'NOT_DETECTED', # OPEN, CLOSED, NOT_DETECTED
    "msg" : "None",
}


# ------------------------
# Máquina de estados
# ------------------------
def process(frame):

    # STEP 0
    if state["step"] == 0: # idle
        state['msg'] = """Pressione 'i' para iniciar, 'm' para mudar o modo ou 'n' para mudar o nome"""
        return frame

    # STEP 1
    if state["step"] == 1: # detectar rosto
        if detect_face(frame, resize_ratio=0.3):
            state['msg'] = "Face detectada. Pisque."
            state["step"] = 2
        else:
            state['msg'] = "Posicione o rosto na camera"

    # STEP 2
    elif state["step"] == 2: # detectar piscada
        current_eye = get_eye_state(frame, resize_ratio=0.3)
        if state['eye_state'] == 'CLOSED' and current_eye== 'OPEN': # detectou piscada
            state['msg'] = "Piscada detectada!"
            state["step"] = 3
        else:
            state['eye_state'] = current_eye
            state['msg'] = f"Aguardando piscada... Olho esta {state['eye_state']}"

    # STEP 3
    elif state["step"] == 3: # capturar face
        if capture_face(frame, state['face_path']):
            state['msg'] = "Face capturada"
            if state["mode"] == "identificar":
                state["step"] = 4
            elif state["mode"] == "cadastrar":
                state["step"] = 5
        else:
            state['msg'] = 'Falha em capturar face, Pressione "R" para reinicar'

    # STEP 4
    elif state["step"] == 4: # Identificar face 
        record = get_record_by_face_picture(state['face_path'])
        if record:
            state['msg'] = f'Face identificada: {record.name} - {record.id}. Pressione "R" para reiniciar'
            state["step"] = 6
        else:
            state['msg'] = f'Face nao identificada. Pressione "R" para reiniciar'
            state["step"] = 6
        

    # STEP 5
    elif state["step"] == 5: # Cadastrar face
        record = register_face(state['face_path'], state['name'])
        if record:
            state['msg'] = f'Face cadastrada: {record.name} - {record.id}. Pressione "R" para reiniciar'
            state["step"] = 6
    
    # STEP 6
    elif state["step"] == 6: # Idle
        pass # estado de espera


    return frame



def write_in_window(frame, text, x, y, font, font_scale, thickness):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # retângulo de fundo
    cv2.rectangle(frame,
                (x, y - text_h - baseline),
                (x + text_w, y + baseline),
                (0, 0, 0),  # cor fundo (preto)
                -1)         # preenchido
    
    cv2.putText(frame, text, (x, y),
                font,
                font_scale, (255, 255, 255), 2)
# ------------------------
# Webcam
# ------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # melhora performance
    frame = cv2.resize(frame, (960, 720))

    frame = process(frame)


    write_in_window(frame, state['msg'], 20,40, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # modo atual
    write_in_window(frame, f"Modo: {state['mode']}", 20,65, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # nome_atual
    write_in_window(frame, f"Nome: {state['name']}", 20,90, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.imshow("Face System", frame)

    key = cv2.waitKey(1) & 0xFF

    # ESC sair
    if key == 27:
        break

    # iniciar
    elif key == ord('i'):
        state["step"] = 1
    elif key == ord('r'):
        state["step"] = 0
        state['eye_state'] = 'NOT_DETECTED'

    # trocar modo
    elif key == ord('m'):
        state["mode"] = "cadastrar" if state["mode"] == "identificar" else "identificar"
    
    elif key == ord('n'):
        i_name += 1
        i_name = i_name%len(list_names)
        state["name"] = list_names[i_name]

cap.release()
cv2.destroyAllWindows()