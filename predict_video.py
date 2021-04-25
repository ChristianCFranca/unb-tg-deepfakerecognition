# Realizar o parse dos nossos argumentos
import argparse
import os

parser = argparse.ArgumentParser(description='Indica se um determinado vídeo é um deepfake.')
parser.add_argument('video_path', type=str, help='Caminho para o vídeo.')
parser.add_argument('--rho', type=float, help='Parâmetro de liberdade.', default=2.75)
parser.add_argument('--follow_along', help='Se deseja que o vídeo apareça em tela durante a extração das imagens.', action="store_true", default=False)
parser.add_argument('--check_every_frame', type=int, help='De quantos em quantos frames se deve verificar o rosto', default=30)
parser.add_argument('--gpu', help='Se deseja que a execução possa escolher uma gpu.', action="store_true", default=False)
args = parser.parse_args()

if not os.path.exists(args.video_path):
    print("Caminho para vídeo não existe, forneça um caminho válido.")
    exit()

# Para adquirir o modelo pré-treinado
from fastai.vision.all import *

# Para iterar pelos vídeos e tornar interativo se desejado
import cv2

# Para extrair os rostos dos frames
from facenet_pytorch import MTCNN

# Útil para realizar operações em vetores
import numpy as np

# Dicionario para contabilizar os FAKE e os REAL
from collections import defaultdict

# Limpa prints extras nas células
from IPython.display import clear_output
# ---------------------------------------------------------------------

# Definimos um device onde os tensores estarão sendo processados
device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
if device == 'cuda:0':
    device = torch.device('cuda:0')
    print(f'Rodando em: {device}, {torch.cuda.get_device_name()}')
else:
    device = torch.device('cpu')
    print(f'Rodando na CPU...')

path_to_learner = Path('./models/final_learner.pkl')
learner = load_learner(path_to_learner, cpu=not args.gpu) # As inferências nas imagens serão feitas pela CPU uma vez que será feito vídeo por vídeo

# Informações para a MTCNN
IMAGE_SIZE = 224
MARGIN = 0
MIN_FACE_SIZE = 90
THRESHOLDS = [0.68, 0.75, 0.80]
POST_PROCESS = False
SELECT_LARGEST = True
KEEP_ALL = False
DEVICE = device

# ----------------------------------

mtcnn = MTCNN(image_size=IMAGE_SIZE,
              margin=MARGIN, 
              min_face_size=MIN_FACE_SIZE, 
              thresholds=THRESHOLDS,
              post_process=POST_PROCESS,
              select_largest=SELECT_LARGEST, 
              keep_all=KEEP_ALL, 
              device=device)

def extract_faces_from_video(video_path, follow_along=False, padding=0, size=-1, resize_factor=0.6, check_every_frame=30):
    
    # Captura o vídeo no path
    try:
        cap = cv2.VideoCapture(str(video_path))
    except:
        print("Ocorreu um erro ao carregar o vídeo. Certifique-se de que é um arquivo de vídeo válido.")
        exit()
    
    # Pega, em inteiros, a quantidade de frames do vídeo
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    faces = []
    
    for i in range(1, v_len + 1):
        success = cap.grab()
        if not success:
            continue
        if  i % check_every_frame == 0:
            success, frame = cap.retrieve()
            if not success:
                continue
        else:
            continue
        
        if success: # Sucesso na leitura
            boxes, _ = mtcnn.detect(Image.fromarray(frame)) # Detecta as imagens. O método detect só aceita numpy arrays
            # Obtém o frame como PIL Image (ele é capturado no formato BGR porém a MTCNN espera no formato RGB)
            frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_new = Image.fromarray(frame_new)
            if boxes is not None: # Só entra se rostos forem detectados
                for box in boxes: # Para cada uma das bouding boxes encontradas em um único frame (a princípio só deve ter uma)
                    box = [int(b) for b in box]
                    # Extrai a face
                    face = frame_new.crop(box=(box[0]-padding, 
                                               box[1]-padding, 
                                               box[2]+padding, 
                                               box[3]+padding))
                    faces.append(PILImage(face))
                    
                    if follow_along:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=[0, 255, 0], thickness=5) # Desenha um retângulo na região do rosto
                        frame = cv2.resize(frame, (int(frame.shape[1]*resize_factor), int(frame.shape[0]*resize_factor)))
                        cv2.imshow('frame', frame)
            
            # Apertar a tecla 'q' para sair do vídeo.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        else:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    return faces

def get_predictions(learner, faces):
    predicts = []
    predicts_dict = defaultdict(lambda: 0)
    for i, face in enumerate(faces):
        res = learner.predict(face)
        print(f"Predição realizada para a face {i+1}")
        predicts.append(res[1].item())
        predicts_dict[res[0]] += 1
    print('-'*100)
    print(F"Resultados individuais: Quantidade de \033[91m FAKES \033[0m: {predicts_dict['FAKE']} | Quantidade de \033[92m REALS \033[0m: {predicts_dict['REAL']}")
    return predicts

def get_final_prediction_from_predictions(predictions, roh=2.75):
    print("-"*100)
    print(f"Utilizando regra com ρ = {roh}...\n")
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    qtd_fakes = np.count_nonzero(predictions == 0)
    qtd_reals = np.count_nonzero(predictions == 1)
    
    return 'FAKE' if qtd_fakes >= roh*qtd_reals else 'REAL'

faces = extract_faces_from_video(args.video_path, follow_along=args.follow_along, check_every_frame=args.check_every_frame)
if len(faces) == 0:
    print("Não foi possível detectar faces humanas no vídeo fornecido.")
    exit()
preds = get_predictions(learner, faces)
final_res = get_final_prediction_from_predictions(preds, roh=args.rho)

color_beg = '\033[92m' if final_res == "REAL" else '\033[91m'
print("\033[94m" + "Resultado Final: " + "\033[0m" + "Vídeo " + color_beg + final_res + "\033[0m")