import numpy as np
import pandas as pd
import cv2

import redis

#Insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
#Time
import time
from datetime import datetime

import os

#Connect to Redis
#hostname = 'redis-18701.c14.us-east-1-2.ec2.redns.redis-cloud.com'
hostname = 'redis-17832.c12.us-east-1-4.ec2.redns.redis-cloud.com'
#port = 18701
port = 17832
#password = 'B4mO6xsgCfXXVWuYxiPxNpb9Z4u4JYLV'
password = 'T4tu2L0UH7BKQiUeQaszrN4Pa0PKUV1l'

r = redis.StrictRedis(host=hostname,
                     port=port,
                     password=password)

#Retrive Data from Database
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'facial_features']
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name', 'Role', 'facial_features']]

#Configure Face Analysis
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

#ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector, name_role=['Name','Role'], thresh=0.5):
    '''
    cosine similarity base search algortihm
    '''
    #Step1: Take the df
    dataframe = dataframe.copy()    
    #Step2: Index face embeding from the dataframe and covert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
        
    #Step3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['Cosine'] = similar_arr
    
    #Step4: Filter the data
    data_filter = dataframe.query(f'Cosine >= {thresh}')
    if len(data_filter) > 0:
        #Step5: Get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['Cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'


    return person_name, person_role

#### Real Time Prediction
#We need to save logs for every 1 mins
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[], current_time=[])

    def saveLogs_redis(self):
        #Step1 - Create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        #Step2 - Drop the duplicates information (distinct name)
        dataframe.drop_duplicates('name', inplace=True)
        #Step3 - Push data to redis database (list)
        #Encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
            
        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)

        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column, name_role=['Name','Role'], thresh=0.5 ):
        #Step0: Find the time
        current_time = str(datetime.now())

        #Step1: Take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        
        #Step2: Use for loop and extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1,y1, x2,y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            #print(person_name, person_role)
            if person_name == 'Desconocido':
                color = (0,0,255)
            else:
                color = (0,255,0)
                
            cv2.rectangle(test_copy, (x1,y1), (x2,y2), color)
        
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1), cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10), cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            
            #Save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    
    def get_embeddings(self, frame):
        #Get results from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            #Put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            #Facial features
            embeddings = res['embedding']
    
        return frame, embeddings

    def save_data_in_redis_db(self, name, role):
        #Validation
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        #If face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'



        #Step1: load 'face_embedding.txt'
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32) #flatten array
        
        #Step2: convert into array
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        #Step3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        #Step4: save this into redis database
        #Redis hashes
        r.hset(name='academy:register', key=key, value=x_mean_bytes)

        #
        os.remove('face_embedding.txt')
        self.reset()

        return True
        
    #Recien creado
    def delete_user_from_redis(name, role):
        """
        Esta función elimina un usuario específico del hash 'academy:register' en Redis.

        :param name: El nombre del usuario que deseas eliminar.
        :param role: El rol del usuario que deseas eliminar.
        :return: True si la eliminación fue exitosa, False si el usuario no se encontró.
        """
        # Crear la clave con el formato 'Nombre@Rol'
        key = f"{name}@{role}"

        # Verificar si la clave existe en Redis
        if r.hexists('academy:register', key):
            # Eliminar la clave del hash
            r.hdel('academy:register', key)
            return True
        else:
            print(f"El usuario {name} con rol {role} no se encontró en Redis.")
            return False


