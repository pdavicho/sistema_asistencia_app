import streamlit as st
from Inicio import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import cv2

# Inicializar en session_state si aún no existe
if 'data_saved' not in st.session_state:
    st.session_state['data_saved'] = False

#st.set_page_config(page_title='Real Time Prediction')
#st.set_page_config(page_title='Detección', page_icon=':👤:', layout='wide')
st.subheader('👤 - Detección')

#Retrive the data from Redis Database
with st.spinner('Esperando BD...'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    #st.dataframe(redis_face_db)

st.success('Datos cargados desde BD')

#Time
waitTime = 10 #Time in sec
setTime = time.time()
savedTime = 0  # To track when data was saved
realtimepred = face_rec.RealTimePred() #Real Time Prediction class

#Real Time Prediction
#Streamlit webrtc

#Callback function
def video_frame_callback(frame):
    global setTime, savedTime

    img = frame.to_ndarray(format="bgr24")
    pred_img = realtimepred.face_prediction(img, redis_face_db,
                                        'facial_features', ['Name', 'Role'], thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    remaining_time = max(0, waitTime - int(difftime))

    # Mostrar contador en el frame
    cv2.putText(pred_img,
                f"Tiempo: {remaining_time}s",
                (10, 30),  # posición (x,y)
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # tamaño de fuente
                (0, 255, 0),  # color BGR (verde)
                2)  # grosor

    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time()
        savedTime = time.time()  # Record when data was saved
        print('Save Data to redis database')
        st.session_state['data_saved'] = True

    # Mostrar mensaje de guardado por 2 segundos
    if time.time() - savedTime < 3:  # If less than 2 seconds have passed since saving
        cv2.putText(pred_img,
                    "Datos guardados!",
                    (int(pred_img.shape[1]/4), int(pred_img.shape[0]/2)),  # centro de la imagen
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),  # verde
                     2)

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePredictions", video_frame_callback=video_frame_callback,
               rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
                )


# Mostrar mensaje si los datos fueron guardados
if st.session_state['data_saved']:
    st.success('Datos registrados en la BD correctamente.')