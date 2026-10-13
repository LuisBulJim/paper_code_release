
## Vistas TTA adicionales (predict_extra_views.py) -- sin reentrenar

Reutiliza los `best.pt` ya guardados en `nested_out/out/runs/<tag>_r{r}_f{k}_{s}/weights/best.pt`
y solo hace inferencia extra sobre las imagenes de val/test de cada fold. Anade estas vistas a los
`.pkl` existentes (guarda un `.bak` antes del primer cambio):

- **`finescale`**: 5 escalas (0.8, 0.9, 1.0, 1.1, 1.2) x flip, en modo `imgsz` (multi-escala real,
  no el resize+letterbox del paper original).
- **`rot`**: rotacion pequena (+-5, +-8 grados por defecto) x flip.
  **Aviso importante**, comprobado con un caso sintetico (rectangulo conocido, sin ruido de
  interpolacion): al rotar sobre el centro de la imagen, una caja lejos del centro se desplaza al
  deshacer la rotacion, y el rectangulo envolvente aproximado pierde precision deprisa: IoU
  recuperado ~0.61 a +-8 grados, ~0.44 a 15 grados (ya no pasa el umbral de emparejamiento 0.5), ~0.37
  a 20 grados. Como la corrosion suele ocupar buena parte del casco cerca de un borde, esta vista
  puede perjudicar mas que ayudar por encima de +-10 grados. Por eso el valor por defecto es
  conservador (5 y 8 grados); dejad que el bootstrap real decida si `tta_rot`/`tta_kitchen` mejoran
  algo, no lo deis por supuesto.
- **`vflip`**: flip vertical solo. El paper original lo excluyo sin mas justificacion que "preservar
  la orientacion de la escena"; aqui se comprueba con datos en vez de suponerlo.
- **`sahi`** (opcional, `--sahi`): tiling con la libreria `sahi`. Esta EXPERIMENTAL y sin probar en
  este entorno (no teniamos forma de verificar que `sahi` funcione con YOLO26 aqui); si falla al
  cargar o al predecir, el script lo salta con un aviso y sigue con todo lo demas sin romperse. Antes
  de lanzarlo en las 45 combinaciones, probadlo en 2-3 imagenes sueltas.

Uso (para los dos protocolos, reutilizando los checkpoints ya entrenados):
```bash
python predict_extra_views.py --out nested_out --tag direct     --scales 0.8 0.9 1.0 1.1 1.2 --rot-angles 5 8
python predict_extra_views.py --out nested_out --tag sequential --scales 0.8 0.9 1.0 1.1 1.2 --rot-angles 5 8
# opcional, tras probarlo en pocas imagenes:
python predict_extra_views.py --out nested_out --tag direct --sahi --sahi-slice 320
```
`--data-root` debe ser el mismo que usasteis en `nested_cv.py` (por defecto `dataset_yolo`).

## Nuevas estrategias en evaluate_cv.py

Pasad `--fine-scales` y `--rot-angles` (los MISMOS valores que usasteis en `predict_extra_views.py`)
para activarlas; si una vista no esta cacheada todavia, la estrategia que la necesita se omite en
silencio (no rompe el resto):

```bash
python evaluate_cv.py --out nested_out --seq-tag sequential --n-boot 1000 --jobs 4 \
    --fine-scales 0.8 0.9 1.0 1.1 1.2 --rot-angles 5 8 --max-views 10
```

- `tta_finescale_m`, `tta_vflip_m`, `tta_rot_m`: variantes sueltas, cada una anade solo un tipo de
  vista sobre el TTA base (escala+flip).
- `tta_kitchen_m`: todo junto (escala fina x flip + rotaciones). Riesgo: con demasiadas vistas, WBF
  puede diluir mas detecciones minoritarias (ver la nota sobre `conf_type='avg'` mas abajo); podria
  no ser la mejor opcion pese al nombre.
- `tta_selected_m`: **seleccion voraz por fold, elegida solo con el conjunto de validacion de ese
  fold** (forward selection: anade la vista que mas sube el F1 de validacion, para hasta 10 vistas
  o hasta que no mejora). Cada fold puede elegir un subconjunto distinto; el test nunca interviene en
  la eleccion. Es la variante mas defendible metodologicamente, aunque no tiene por que ser la mejor
  en numeros: puede que la busqueda converja a pocas vistas si el conjunto de validacion (unas 35
  imagenes) no tiene senal suficiente para distinguir combinaciones parecidas.
- `tta_imgsz_m_maxconf`, `tta_paper_m_maxconf`, `ens_ml_11_maxconf`: mismas cajas, pero puntuando el
  AP50 con `conf_type='max'` en vez de `'avg'`. **No cambian Precision/Recall/F1** (eso se decide
  antes, por el filtro a 0.25 de cada vista); solo pueden mover el ranking del AP50.
  Comprobado analiticamente: `box_and_model_avg` y `absent_model_aware_avg` dan el MISMO resultado
  que `avg` en este montaje (cada vista aporta como mucho una caja por cluster), asi que no se
  incluyen como variantes propias -- no aportarian nada nuevo.

Las comparaciones bootstrap ahora incluyen cada variante nueva frente a `single_m`, frente a
`tta_imgsz_m` (el mejor del informe anterior) y frente a `tta_paper_m` (el original del paper), y
frente a los dos ensembles.

## bench_latency.py: aislamiento de memoria por proceso

La version anterior podia arrastrar memoria CUDA de una estrategia a la siguiente dentro del MISMO
proceso, inflando `weights_MB`/`peak_MB` de forma creciente y poco fiable (visto en una ejecucion
real: 110->190->200->373 MB, sin relacion con el tamano real de los pesos). Ahora cada estrategia se
relanza en un proceso Python nuevo (usa `--only <estrategia>` internamente); la latencia
(`mean_ms`/`p95_ms`) no deberia haber cambiado, pero la memoria ahora es fiable. Mismo uso de
siempre:
```bash
python bench_latency.py --n .../n/weights/best.pt --m .../m/weights/best.pt --l .../l/weights/best.pt \
    --images dataset_yolo/images/test
```

## Agrupacion de folds por video/secuencia (check_filename_order.py + build_groups.py)

LIACI son ~1893 imagenes extraidas de unos 16-17 videos de inspeccion; sin agrupar por video,
fotogramas del mismo video pueden caer uno en train y otro en test del mismo fold, inflando los
resultados.

**Hallazgo clave (de los nombres reales que nos pasasteis):** los ficheros no son `image_0001.jpg`
... `image_0261.jpg` secuenciales -- son `image_0037.jpg`, `image_0297.jpg`, `image_1858.jpg`, etc.,
con indices dispersos hasta ~1890+, justo el tamano del release completo de LIACI (~1893 imagenes).
Esto quiere decir que el numero es probablemente el **indice global original de LIACI**, no una
renumeracion de este proyecto. Si LIACI numero sus imagenes en orden de extraccion por video (lo
habitual), imagenes con indices CERCANOS en valor -- aunque no sean consecutivas dentro de nuestro
subconjunto de 261 -- son candidatas a venir del mismo video. Con los 33 indices del test que nos
pasasteis ya se ve una estructura prometedora: huecos pequenos (2-8) dentro de lo que parecen rafagas,
y huecos grandes (68-197) entre ellas.

**Por eso el metodo por defecto ahora es `idgap`**, que agrupa por esos huecos numericos, sin cargar
ni comparar ninguna imagen. Es mas rapido y, con esta pista, mas fiable que los dos metodos visuales
anteriores (que siguen disponibles como `--method sequential` / `--method graph` por si `idgap` no
aplica, p. ej. si los nombres no llevan un numero, o si el numero resulta no llevar señal real).

**Paso 0 -- diagnostico rapido (opcional con idgap, pero gratis y confirma la hipotesis):**
```bash
python revision_nested_cv/check_filename_order.py --data-root dataset_yolo
```

**Paso 1 -- agrupar por huecos de indice (el metodo recomendado):**
```bash
python revision_nested_cv/build_groups.py cluster --data-root dataset_yolo \
    --out groups.csv --report groups_report.csv --method idgap
```
Imprime un HISTOGRAMA de todos los huecos entre IDs consecutivos (ordenados por numero) de las 261
imagenes -- es el dato mas importante: si aparece la misma estructura bimodal que ya vimos en la
muestra de 33 (huecos pequenos muy frecuentes + huecos grandes claramente separados), el metodo
deberia funcionar bien de forma directa. Revisad tambien:
- el numero de grupos (deberia acercarse a ~16-17);
- `groups_report.csv`, columna `id_gap_to_prev_by_id`, para confirmar a ojo que los cortes caen en
  los huecos grandes y no en mitad de una rafaga de huecos pequenos.

Si el histograma NO muestra una separacion clara (huecos pequenos y grandes mezclados sin hueco en
medio), el numero probablemente no es el indice global de LIACI o no lleva señal de video, y hay que
probar `--method graph` (agrupamiento por similitud visual, componentes conexas sobre un hash
perceptual, tambien imprime su propio histograma) como alternativa; si ninguno de los dos separa
limpio, lo correcto es reportarlo como limitacion sin resolver en el paper en vez de forzar un numero
de grupos arbitrario.

Parametros ajustables de `idgap` (todos con valores por defecto razonables, pero mirad el histograma
antes de confiar en ellos): `--k` (cuanto mayor, menos cortes), `--min-id-gap` (suelo absoluto por
debajo del cual un hueco nunca es un corte, por defecto 15), `--min-periods` (puntos de historial
minimos antes de activar el chequeo adaptativo, por defecto 3 -- evitad bajarlo de 2).

**Validado con un caso sintetico que reproduce el patron real** (rafagas de huecos pequenos entre
"videos" separados por huecos grandes, generado a partir de los propios numeros que nos pasasteis):
recupera los 7 "videos" sinteticos exactamente, en ambas direcciones (ningun video partido en dos
grupos, ningun grupo mezclando dos videos). Tambien corregimos un bug real que encontramos con este
mismo test: una version anterior daba un "pase libre" (sin comprobar nada) a los primeros huecos tras
cada corte, asi que un salto grande que caia justo ahi se colaba sin marcarse -- esto fusiono 2 de los
7 videos sinteticos hasta que lo arreglamos.

**Metodos alternativos** (`--method graph`, `--method sequential`): documentados en el codigo; usad
`graph` si `idgap` no aplica (nombres sin numero, o numero sin señal real segun el histograma).
`sequential` (el metodo original, basado en el ORDEN de los nombres, no en su valor) tiene una
debilidad conocida y no arreglada con secuencias de deriva visual muy continua y sin cortes duros; no
es el metodo recomendado para este dataset.

**Paso 2 -- relanzar con la agrupacion, solo cuando tengais un `groups.csv` en el que confieis:**
```bash
GROUPS_CSV=groups.csv bash revision_nested_cv/run_revision.sh direct
GROUPS_CSV=groups.csv bash revision_nested_cv/run_revision.sh sequential
bash revision_nested_cv/run_revision.sh verify      # <-- CONFIRMAD aqui que de verdad reentreno
bash revision_nested_cv/run_revision.sh extra
bash revision_nested_cv/run_revision.sh eval
```

**IMPORTANTE -- bug real que nos paso DOS veces y ya esta arreglado, pero verificadlo igualmente:**
si reusais el mismo `--out` (por defecto, `revision_nested_cv/out`) que ya teniais de una tanda sin
agrupar, una version anterior de `nested_cv.py` se SALTABA el reentrenamiento de cada fold porque el
fichero de predicciones de ese (repeticion, fold, tamano) ya existia -- sin comprobar si el split
con el que se habia entrenado seguia siendo el mismo. Resultado: los resultados salian identicos a
los de antes, no porque la agrupacion no tuviera efecto, sino porque nunca se llego a aplicar. Ya esta
arreglado (cada `.pkl` guarda una huella del split con el que se entreno, y si no coincide con el
split actual, se reentrena aunque el fichero exista). Aun asi, ejecutad siempre:
```bash
bash revision_nested_cv/run_revision.sh verify
```
inmediatamente despues de `direct`/`sequential`, ANTES de gastar tiempo en `extra`/`eval`. Os dira,
sin entrenar nada, cuantos ficheros estan al dia, cuantos desactualizados (hace falta reentrenar) y
cuantos sin entrenar. Si dice "0 desactualizados, 0 sin entrenar", es que de verdad se aplico el
reentrenamiento agrupado.

