# mammo_prep

Paquete de utilidades para preprocesado de imágenes de mamografía.

## Instalación

Desde la carpeta raíz del proyecto (donde está `pyproject.toml`):

```bash
pip install -e .
```

El flag `-e` (editable) hace que cualquier cambio que hagas en el código
se aplique automáticamente sin necesidad de reinstalar.

## Uso en notebooks

```python
# Importar módulos completos
from mammo_prep import io, normalize, artifacts, viz

# O importar funciones concretas
from mammo_prep.io import load_dicom
from mammo_prep.normalize import clahe_normalize
from mammo_prep.artifacts import crop_to_breast
from mammo_prep.viz import plot_comparison

# Ejemplo básico
img = load_dicom("ruta/a/imagen.dcm")
img_norm = clahe_normalize(img)
img_crop = crop_to_breast(img_norm)
plot_comparison(img, img_crop, titles=["Original", "Procesada"])
```

## Módulos

| Módulo | Contenido |
|---|---|
| `io.py` | Carga y guardado de imágenes (DICOM, PNG, JPG) |
| `normalize.py` | Normalización y estandarización de píxeles |
| `artifacts.py` | Eliminación de fondo, artefactos y orientación |
| `viz.py` | Visualización e histogramas |
