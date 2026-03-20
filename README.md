# Marketing AI - listo para Render

## Archivos
- `app.py`: app web en Streamlit
- `requirements.txt`: dependencias
- `render.yaml`: configuración para desplegar en Render

## Cómo desplegar
1. Sube estos archivos a un repositorio de GitHub.
2. Agrega tu archivo `sales_data_sample.csv` en la raíz del proyecto.
3. En Render, crea un **New Web Service** desde tu repo.
4. Si Render detecta `render.yaml`, usará esta configuración automáticamente.
5. Espera a que termine el build y abre la URL pública.

## Comando local
```bash
streamlit run app.py
```

## Nota
El código original venía de Colab, así que se eliminaron partes como `drive.mount()` para que funcione en una web normal.
