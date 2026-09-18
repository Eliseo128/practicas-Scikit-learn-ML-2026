¡Por supuesto! Trabajar con **Scikit-learn** en cuadernos de **Jupyter (`.ipynb`)** dentro de VS Code es ideal para la enseñanza, ya que permite ver las predicciones, tablas de datos e impresiones en pantalla justo debajo de cada bloque de código.

A continuación, muestro el procedimiento completo para configurar el entorno e interactuar con los 5 cuadernos interactivos.

---

### **Paso 1: Configuración del entorno virtual e instalación de Jupyter**

Abre la terminal integrada en VS Code dentro de la carpeta del proyecto y ejecuta los siguientes comandos:

1. **Entrar a la carpeta del proyecto:**
```bash
cd practica8-ml-scikit-learn

```


2. **Asegurarte de que el entorno `.venv8` esté activado:**
* **Windows (CMD):** `.venv8\Scripts\activate`
* **Windows (PowerShell):** `.venv8\Scripts\Activate.ps1`
* **macOS/Linux:** `source .venv8/bin/activate`


3. **Instalar Scikit-learn, Jupyter e ipykernel:**
```bash
pip install scikit-learn jupyter ipykernel

```


4. **Vincular el entorno virtual a Jupyter Notebook:**
Este comando registra el entorno `.venv8` para que VS Code lo detecte como un núcleo (*kernel*) disponible:
```bash
python -m ipykernel install --user --name=.venv8 --display-name "Python (.venv8)"

```


5. **Seleccionar el Kernel en VS Code:**
* Abre cualquiera de los archivos `.ipynb` en VS Code.
* Haz clic en **Select Kernel** (o **Seleccionar Kernel**) en la esquina superior derecha de la ventana del cuaderno.
* Elige **Python Environments...** y selecciona la opción que dice **Python (.venv8)**.



---

### **Paso 2: Estructura de carpetas y archivos `.ipynb**`

Una vez creados los cuadernos desde el explorador de archivos de VS Code, la estructura quedará de la siguiente manera:

```text
practica8-ml-scikit-learn/
│
├── .venv8/                      # Entorno virtual aislado
├── practica8a-0777.ipynb        # Cuaderno 1: Regresión Lineal
├── practica8b-0777.ipynb        # Cuaderno 2: Clasificación Logística
├── practica8c-0777.ipynb        # Cuaderno 3: Árbol de Decisión
├── practica8d-0777.ipynb        # Cuaderno 4: Agrupamiento K-Means
└── practica8e-0777.ipynb        # Cuaderno 5: K-Vecinos más Cercanos (KNN)

```

---

### **Paso 3: Códigos de Python para las celdas del Notebook**

En cada archivo `.ipynb`, crea una celda de código (Code) e ingresa el código correspondiente. Al ejecutar la celda (presionando `Shift + Enter` o el botón de *Play* a la izquierda de la celda), el resultado de Scikit-learn se desplegará en tiempo real.

---

#### **1. `practica8a-0777.ipynb` — Regresión Lineal Simple**

> **Explicación didáctica:** Predice un número continuo (calificación del 0 al 10) según las horas de estudio invertidas.

```python
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento: [Horas de estudio] (Matriz 2D)
X_train = [[1], [2], [3], [5], [6], [8], [9]]

# Respuestas conocidas: Calificación obtenida
y_train = [5.0, 5.5, 6.2, 7.5, 8.1, 9.2, 9.8]

# 1. Crear el modelo
modelo = LinearRegression()

# 2. Entrenar el modelo
modelo.fit(X_train, y_train)

# 3. Hacer una predicción para un alumno que estudió 7 horas
horas_nuevas = [[7]]
prediccion = modelo.predict(horas_nuevas)

print("--- PREDICCIÓN DE CALIFICACIÓN ---")
print(f"Si estudias {horas_nuevas[0][0]} horas, tu calificación estimada es: {prediccion[0]:.2f}")

```

---

#### **2. `practica8b-0777.ipynb` — Clasificación Logística**

> **Explicación didáctica:** Evalúa dos variables (horas de estudio y % de asistencia) para clasificar al alumno en una categoría binaria: *"En Riesgo"* o *"A Salvo"*.

```python
from sklearn.linear_model import LogisticRegression

# Datos: [Horas de estudio, Asistencias en %]
X_train = [
    [1, 50],   # Estudió 1h, 50% asistencias
    [2, 60],   # Estudió 2h, 60% asistencias
    [5, 85],   # Estudió 5h, 85% asistencias
    [7, 90],   # Estudió 7h, 90% asistencias
    [8, 95]    # Estudió 8h, 95% asistencias
]

# Etiquetas: 0 = En Riesgo, 1 = A Salvo
y_train = [0, 0, 1, 1, 1]

# 1. Crear y entrenar el modelo
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 2. Evaluar a un nuevo estudiante (3 horas de estudio, 70% de asistencia)
nuevo_estudiante = [[3, 70]]
resultado = modelo.predict(nuevo_estudiante)

etiqueta = "A Salvo" if resultado[0] == 1 else "En Riesgo"

print("--- EVALUACIÓN DE RIESGO ACADÉMICO ---")
print(f"Estudiante con 3h de estudio y 70% asistencia se clasifica como: {etiqueta}")

```

---

#### **3. `practica8c-0777.ipynb` — Árbol de Decisión (Filtro Spam)**

> **Explicación didáctica:** Aplica reglas condicionales automáticas basadas en las características del correo (número de links y si incluye la palabra "Gratis").

```python
from sklearn.tree import DecisionTreeClassifier

# Características del correo: [Número de links, Contiene 'Gratis' (1=Sí, 0=No)]
X_train = [
    [0, 0],  # Correo normal de un amigo
    [5, 1],  # Oferta de premio
    [1, 0],  # Tarea escolar con 1 enlace
    [7, 1],  # Oferta sospechosa
    [0, 1]   # Mensaje con la palabra gratis pero sin enlaces
]

# Etiquetas de texto
y_train = ["Normal", "Spam", "Normal", "Spam", "Spam"]

# 1. Crear el árbol de decisión y entrenar
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# 2. Clasificar un correo nuevo con 4 links y que contiene la palabra "Gratis" (1)
correo_nuevo = [[4, 1]]
prediccion = modelo.predict(correo_nuevo)

print("--- FILTRO AUTOMÁTICO DE CORREO ---")
print(f"El correo analizado fue clasificado como: {prediccion[0]}")

```

---

#### **4. `practica8d-0777.ipynb` — Agrupamiento K-Means (Aprendizaje No Supervisado)**

> **Explicación didáctica:** Algoritmo que agrupa automáticamente los datos en $K=2$ conjuntos de clientes sin tener etiquetas previas.

```python
from sklearn.cluster import KMeans

# Datos: [Monto gastado en la cafetería ($MXN), Visitas al mes]
clientes = [
    [50, 2],    # Comprador ocasional
    [60, 3],    # Comprador ocasional
    [500, 15],  # Cliente frecuente
    [550, 18],  # Cliente frecuente
    [480, 12],  # Cliente frecuente
    [40, 1]     # Comprador ocasional
]

# 1. Configurar K-Means para encontrar 2 grupos (K=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(clientes)

# 2. Obtener la asignación de grupos de cada cliente
etiquetas_grupos = kmeans.labels_

print("--- SEGMENTACIÓN AUTOMÁTICA DE CLIENTES ---")
for i, cliente in enumerate(clientes):
    print(f"Cliente {i+1} {cliente} -> Grupo asignado: {etiquetas_grupos[i]}")

```

---

#### **5. `practica8e-0777.ipynb` — K-Vecinos más Cercanos (KNN)**

> **Explicación didáctica:** Sistema de recomendación que busca los 3 de deportistas más parecidos (en peso y estatura) para sugerir la disciplina más adecuada.

```python
from sklearn.neighbors import KNeighborsClassifier

# Datos de deportistas: [Estatura en cm, Peso en kg]
X_train = [
    [160, 55],  # Perfil A
    [165, 58],  # Perfil A
    [185, 85],  # Perfil B
    [190, 90],  # Perfil B
    [175, 70]   # Perfil C
]

# Deporte asignado
y_train = ["Gimnasia", "Gimnasia", "Básquetbol", "Básquetbol", "Fútbol"]

# 1. Crear el clasificador KNN con K=3 vecinos
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# 2. Recomendar deporte a un alumno de 188 cm y 88 kg
nuevo_alumno = [[188, 88]]
deporte_recomendado = modelo.predict(nuevo_alumno)

print("--- SISTEMA DE RECOMENDACIÓN DEPORTIVA ---")
print(f"Para una estatura de {nuevo_alumno[0][0]}cm y {nuevo_alumno[0][1]}kg, se recomienda: {deporte_recomendado[0]}")

```
