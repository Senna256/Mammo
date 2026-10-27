# mammo_prep

Preprocessament de mamografies en format DICOM per a models de detecció de lesions.

Amb una sola crida, una mamografia DICOM es converteix en una imatge de 1024 × 1024 × 3 a punt per entrar al model:

```python
from mammo_prep import preprocess_mammogram

image, transform = preprocess_mammogram("path/to/image.dicom")
```

## Instal·lació

Des d'aquesta carpeta (on hi ha `pyproject.toml`):

```bash
pip install -e .
```

Amb `-e` (mode editable), els canvis al codi s'apliquen sense haver de reinstal·lar el paquet.

Dependències: `pydicom`, `opencv-python`, `numpy` i `matplotlib`.

## Exemple

`examples/demo_preprocess.py` mostra pas a pas què es fa a una mamografia, des de la imatge original fins a l'entrada del model. Desa una figura amb cinc panells:

1. DICOM original, sense cap processament.
2. DICOM amb windowing.
3. Imatge redimensionada, amb el retall de la mama.
4. El retall.
5. Entrada del model.

Si es passen caixes de lesions, es dibuixen a cada pas.

```bash
python examples/demo_preprocess.py --dicom image.dicom --box 2492 1878 2596 1974 --out demo.png
```

`--box x1 y1 x2 y2` és una lesió en píxels del DICOM original. Es pot repetir per indicar-ne diverses.

## Preprocessament

### Passos

`preprocess_mammogram()` aplica aquests passos en ordre:

| # | Pas | Funció |
|---|---|---|
| 1 | Llegeix el DICOM | `load_dicom` |
| 2 | Si és MONOCHROME1 (fons clar), l'inverteix com a `max − imatge` perquè el teixit quedi clar | `window_dicom` |
| 3 | Aplica el windowing `breast_tissue` (percentils 25-95 dels píxels > 0, amplada × 1,5) amb la funció VOI `LINEAR`, i converteix a uint8 | `preprocess_window` |
| 4 | Redimensiona fins que el costat llarg fa 1024 px, mantenint la proporció | `resize_keep_aspect` |
| 5 | Retalla la mama | `detect_breast_roi` |
| 6 | Passa a 3 canals i la col·loca centrada en un quadrat de 1024 × 1024 amb fons negre | `resize_and_pad` |

### Com es troba la mama (pas 5)

`detect_breast_roi` és **l'únic** mètode de retall del preprocés:

1. Aplica un llindar: els píxels > 5 es consideren mama.
2. Fa un tancament morfològic (el·lipse de 31 px) per omplir els buits del teixit gras.
3. Fa una obertura (el·lipse de 31 px) per eliminar etiquetes de text i marcadors petits.
4. Es queda amb el rectangle que envolta el contorn més gran.

Si no troba cap regió raonable, conserva tota la imatge. Una regió no és raonable si fa menys del 5 % de l'amplada o de l'alçada. Això passa sobretot quan el fons de la imatge és gris en lloc de negre: no es perd cap lesió, però la mama queda més petita dins la imatge final.

### Caixes de lesions

Les caixes s'expressen com a `(x1, y1, x2, y2)` en píxels.

- `transform_boxes` passa les caixes del DICOM original a la imatge preprocessada (per exemple, el ground truth). Les retalla al retall de la mama i elimina les que hi queden fora.
- `inverse_transform_boxes` fa el camí contrari (per exemple, les prediccions del model). Si una caixa és sencera dins del retall, l'anada i tornada és exacta.

```python
from mammo_prep import preprocess_mammogram, transform_boxes, inverse_transform_boxes

image, transform = preprocess_mammogram(path)
gt_in_image = transform_boxes(gt_boxes, transform)          # DICOM -> imatge del model
pred_in_dicom = inverse_transform_boxes(predictions, transform)  # imatge del model -> DICOM
```

### Paràmetres

L'únic paràmetre configurable és `img_size`, la mida del quadrat final (per defecte 1024).

La resta de valors són fixos:
- mètode de windowing;
- llindar de 5;
- nucli de 31 px;
- interpolació lineal.

El llindar i el nucli estan ajustats per a imatges de 1024 px. Amb un altre `img_size` el retall funciona, però pot ser menys precís.

## Referència de funcions

Totes es poden importar directament amb `from mammo_prep import <funció>`.

### `pipeline.py`: preprocessament complet

#### `preprocess_mammogram(dicom_path, img_size=1024)`
Aplica tot el preprocessament (passos 1-6).
- **Retorna** `(image, transform)`:
  - `image`: uint8 de mida `(img_size, img_size, 3)`. Els tres canals contenen la mateixa imatge en grisos.
  - `transform`: diccionari amb els passos geomètrics aplicats:

| Clau | Contingut |
|---|---|
| `original_size` | Mida del DICOM `(w, h)` |
| `resized_size` | Mida després del pas 4 `(w, h)` |
| `roi` | Retall de la mama `(x1, y1, x2, y2)` dins la imatge redimensionada |
| `scale` | Escala del pas 6 |
| `pad` | Farciment `(pad_x, pad_y)` del pas 6 |
| `img_size` | Mida del quadrat final |

#### `transform_boxes(boxes, transform)`
Converteix caixes en píxels del DICOM original a píxels de la imatge preprocessada.
- Les caixes es retallen al retall de la mama, i les que hi queden amb ≤ 1 px s'eliminen.
- **Retorna** `np.ndarray float32 (N, 4)`.

#### `inverse_transform_boxes(boxes, transform)`
Converteix caixes de la imatge preprocessada a píxels del DICOM original.
- **Retorna** `np.ndarray float32 (N, 4)`.

#### `window_dicom(dicom_path)`
Aplica els passos 1-3: lectura, inversió MONOCHROME1 i windowing `breast_tissue`.
- **Retorna** uint8 `(H, W)` a la resolució original.

### `io.py`: lectura

#### `load_dicom(path)`
- **Retorna** `(image, ds)`:
  - `image`: el `pixel_array` en float32;
  - `ds`: el dataset de pydicom amb les metadades.

### `windowing.py`: finestra d'intensitats

#### `preprocess_window(image, dicom_dataset=None, method="dicom", voi_func=None, exclude_background=True, output_dtype=np.uint8)`
Calcula una finestra i l'aplica.
- `image`: matriu de píxels. Si és MONOCHROME1, ja ha d'estar invertida.
- `method`: amb `"dicom"` fa servir la finestra de la capçalera i cal passar `dicom_dataset`. Amb qualsevol altre mètode de la taula de sota, la finestra es calcula a partir de la imatge.
- `voi_func`: substitueix la funció VOI del DICOM. Valors: `"LINEAR"`, `"LINEAR_EXACT"`, `"SIGMOID"`.
- `output_dtype`: amb `np.uint8` (per defecte) arrodoneix a [0, 255]. Amb qualsevol altre tipus retorna els valors sense arrodonir.

#### `calculate_windowing(image, method="percentile_2_98", exclude_background=True)`
Calcula `(center, width)` a partir de la imatge.

| Mètode | Finestra |
|---|---|
| `percentile_1_99`, `percentile_2_98`, `percentile_5_95` | Entre els dos percentils |
| `breast_tissue` | Percentils 25-95, amplada × 1,5 |
| `statistical` / `statistical_wide` | Mitjana ± 2 / 3 desviacions estàndard |
| `full_range` | Del mínim al màxim |
| `histogram_peak` | Centrada en el pic de l'histograma, amplada 1,75 × FWHM |

Amb `exclude_background=True` s'ignoren els píxels ≤ 0.

#### `calculate_all_methods(image, exclude_background=True)`
Retorna `{mètode: (center, width)}` per a tots els mètodes. Serveix per comparar finestres.

#### `get_windowing_for_dicom(image, dicom_dataset, method="dicom", exclude_background=True)`
Retorna els paràmetres de la finestra. Amb `method="dicom"` els llegeix de la capçalera; amb qualsevol altre mètode, els calcula.

Claus del diccionari:
- `window_center`, `window_width`;
- `rescale_intercept`, `rescale_slope`;
- `voi_lut_function`;
- `source`: `"dicom"` o `"calculated"`.

#### `get_dicom_voi_lut_params(dicom_dataset)`
Retorna la finestra guardada a la capçalera DICOM. Si la capçalera no en té cap de vàlida, la calcula amb `breast_tissue`.

#### `apply_windowing(image, window_width, window_center, voi_func="LINEAR", y_min=0, y_max=255)`
Aplica una finestra segons la definició VOI LUT del DICOM i retorna valors float32.
- `LINEAR` segueix l'estàndard DICOM (center − 0,5 i width − 1).
- `SIGMOID` no retalla els valors a [y_min, y_max].

#### `should_invert_monochrome1(ds)`
Retorna `True` si el DICOM és MONOCHROME1.

#### `normalize_photometric(image, ds)`
Inverteix les imatges MONOCHROME1 com a `max + min − imatge`.

### `artifacts.py`: retall, mida i orientació

#### `detect_breast_roi(img)`
Retorna el rectangle `(x1, y1, x2, y2)` de la mama; el mètode s'explica a [Com es troba la mama](#com-es-troba-la-mama-pas-5).
- `x2` i `y2` són exclusius: la imatge es retalla amb `img[y1:y2, x1:x2]`.
- L'entrada ha de ser una imatge uint8 2D amb windowing aplicat. Si no ho és, llança `ValueError`.

#### `resize_keep_aspect(img, max_size)`
Redimensiona mantenint la proporció fins que el costat llarg fa `max_size`, amb interpolació lineal.
- **Retorna** `(imatge, scale)`.

#### `resize_and_pad(img, img_size)`
Col·loca una imatge de 3 canals centrada en un quadrat de `img_size` × `img_size` amb fons negre.
- **Retorna** `(imatge, scale, pad_x, pad_y)`.
- Un punt `(x, y)` de l'entrada va a `(x·scale + pad_x, y·scale + pad_y)`.

#### `crop_breast(img, mask=None)`
Retalla la imatge al rectangle d'una màscara. Si no se li passa cap màscara, la calcula amb `get_breast_mask`.
- **Retorna** `(retall, màscara retallada)`.
- No forma part del preprocessament.

#### `resize_long_side(img, target_size=1536, interpolation=cv2.INTER_AREA)`
Redimensiona fins que el costat llarg fa `target_size`.

#### `pad_to_square(img)`
Afegeix zeros fins a fer la imatge quadrada. El costat del quadrat és el costat més llarg de la imatge.

#### `flip_to_standard(img, ds)`
Gira horitzontalment les imatges amb `ImageLaterality == "R"`.

### `normalize.py`: intensitats i màscara

#### `normalize_to_uint8(img)`
Retalla als percentils 1-99 i escala a uint8 [0, 255].

#### `apply_clahe(img, clip_limit=2.0, title_grid_size=(8, 8))`
Aplica CLAHE (contrast local) a una imatge uint8.

#### `get_breast_mask(img, min_area=2000)`
Calcula una màscara de la mama (0/255) amb el llindar d'Otsu i es queda amb el component connex més gran.
- Accepta intensitats DICOM crues.
- No forma part del preprocessament.

### `viz.py`: visualització

#### `show(img, title="")`
Mostra una imatge en grisos.

#### `plot_comparison(images, titles=None, figsize=(15, 5))`
Mostra diverses imatges en fila per comparar-les.
