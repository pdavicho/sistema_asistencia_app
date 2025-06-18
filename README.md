# 🧑‍🏫 Sistema de Asistencia con Reconocimiento Facial (InsightFace + Redis + Streamlit)

Este proyecto implementa un **sistema inteligente de asistencia** basado en **reconocimiento facial en tiempo real**, utilizando la potente librería **InsightFace** con el modelo `buffalo_sc`. Combina una base de datos en memoria con **Redis** para almacenar los embeddings y una interfaz moderna construida con **Streamlit**.

## 🚀 Características principales

- 🧠 Reconocimiento facial preciso con `buffalo_sc` (InsightFace)
- 📸 Registro de usuarios con al menos 200 imágenes por persona
- 🧬 Generación y almacenamiento de embeddings en Redis
- 🧑‍🎓 Asociación de nombre y rol al vector facial
- 🕵️ Detección en tiempo real y registro automático de asistencia
- 📊 Página de reportes con historial completo de asistencias
- 📦 Sistema completamente modular y escalable
- 🌐 Interfaz web amigable construida con Streamlit

---

## 🧰 Tecnologías utilizadas

- [InsightFace](https://github.com/deepinsight/insightface)  
- Modelo preentrenado: `buffalo_sc`  
- Streamlit  
- Redis (modo local o contenedor Docker)  
- Python 3.10
- Numpy, OpenCV, pandas


## 📌 Funcionalidades

### 🔹 Página: Registro de Usuarios

* Captura automática de 200 imágenes por persona
* Extracción de embeddings usando `buffalo_sc`
* Almacenamiento de vectores faciales en Redis
* Asociación con nombre completo y rol (e.g., estudiante, docente)

### 🔹 Página: Asistencia

* Reconocimiento facial en tiempo real con cámara
* Detección de identidad mediante comparación de embeddings
* Registro automático de fecha, hora y persona reconocida

### 🔹 Página: Reportes

* Visualización y exportación de registros de asistencia
* Filtro por fecha, persona o rol
* Descarga en formato CSV

---

## 🤖 Créditos

Desarrollado por [David Minango](https://github.com/pdavicho).
Este sistema fue diseñado como solución práctica y funcional para **control de asistencia en instituciones educativas o laborales**.

---

## 📜 Licencia

MIT License – libre para uso, modificación y distribución con atribución.

```

