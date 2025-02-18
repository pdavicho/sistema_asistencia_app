import streamlit as st
from PIL import Image  # Add this import at the top

st.set_page_config(page_title='Reconocimiento Facial', page_icon=':👤:', layout='wide')

# Create a sidebar and add the image
with st.sidebar:
    image = Image.open('ister.png')  # Replace with your image path
    st.image(image, width=200)  # Smaller width for sidebar

st.header('Sistema de Asistencia usando Reconocimiento Facial')

with st.spinner('Cargando Modelo y conectandose a la BD...'):
    import face_rec

st.success('Sistema Listo')
#st.success('Model loaded sucessfully')
#st.success('Redis db sucessfully connected')