import os
import pickle
from deepface import DeepFace
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

REGISTER_FILE= 'face_id_register.pkl'

class Record:
    def __init__(self, id, name, embedding):
        self.id = id
        self.name = name
        self.embedding = embedding
    def __repr__(self):
        return f'Record id: {self.id} - {self.name}'

def register_face(path_imageFace, name):
    pass
    
    # process face embedding
    embedding = embed_face_image(path_imageFace)
    
    # check if face is in file already


    # append face ebedding with ID and name
    registry = get_all_records()
    max_id = max(record.id for record in registry)
    id = max_id + 1
    record = Record(id, name, embedding)
    load_record(record)
    return True

def clear_registry():
    os.remove(REGISTER_FILE)

def similarity(emb_1, emb_2):
    if len(emb_1) != len(emb_2):
        return 0 
    aemb_1 = np.array(emb_1)
    aemb_2 = np.array(emb_2)
    similarity = np.dot(aemb_1, aemb_2) / (
        np.linalg.norm(aemb_1) * np.linalg.norm(aemb_2)
    )
    return similarity

def load_record(record):
    registry = get_all_records()
    registry.append(record)
    with open(REGISTER_FILE, 'wb') as f:
            pickle.dump(registry, f)

def get_all_records():
    path_registry = Path(REGISTER_FILE)
    
    if not path_registry.exists():
        with open(REGISTER_FILE, 'wb') as f:
            empty_record = Record(0, '', [])
            registry = [empty_record]
            pickle.dump(registry, f) # empty record for initialization

    with open(REGISTER_FILE, 'rb') as f:
        registry = pickle.load(f)
    return registry

def get_record_by_id(id):
    # read registry 
    registry = get_all_records()
    records = [record for record in registry if id == record.id]
    if len(records) < 1:
        print(f'ERROR - Registro de ID {id} não encontrado no registry')
        return False
    return records[0]

def get_record_by_emb(embeding, treshhold = 0.6):
    registry = get_all_records()
    for record in registry:
        if similarity(embeding, record.embedding) >= treshhold:
            return record
    return False # no record found

def get_records_by_name(name):
    # read registry 
    registry = get_all_records()
    records = [record for record in registry if name in record.name]
    return records

def embed_face_image(path_imageFace):
    embedding = DeepFace.represent(path_imageFace, model_name='ArcFace', enforce_detection=False)[0]['embedding']
    return embedding
