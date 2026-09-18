¡Hola! Qué gusto saludarte de nuevo. Ahora daremos un paso enorme en nuestro camino de aprendizaje: pasaremos de visualización a **Machine Learning en la vida real** utilizando **Scikit-learn** (`sklearn`), la librería estándar para la creación de modelos predictivos y de clasificación.

Como tu docente didáctico, he adaptado 5 ejemplos sencillos con contextos cotidianos para estudiantes de preparatoria.

---

### **Paso 1: Configuración del entorno de trabajo en la terminal**

Abre la terminal integrada en VS Code y ejecuta los siguientes comandos paso a paso:

1. **Crear la carpeta del proyecto y entrar en ella:**
```bash
mkdir practica8-ml-scikit-learn
cd practica8-ml-scikit-learn

```


2. **Crear el entorno virtual `.venv8`:**
* En Windows: `python -m venv .venv8`
* En macOS/Linux: `python3 -m venv .venv8`


3. **Activar el entorno virtual:**
* En Windows (CMD): `.venv8\Scripts\activate`
* En Windows (PowerShell): `.venv8\Scripts\Activate.ps1`
* En macOS/Linux: `source .venv8/bin/activate`


4. **Instalar Scikit-learn (y dependencias necesarias):**
```bash
pip install scikit-learn

```


5. **Seleccionar el intérprete de Python en VS Code:**
* Presiona `Ctrl + Shift + P` (o `Cmd + Shift + P` en macOS).
* Escribe **Python: Select Interpreter**.
* Selecciona la opción que apunta a `.venv8` (debe decir `('.venv8': venv)`).



---

### **Paso 2: Estructura de carpetas y archivos**

Al crear los archivos Python dentro del proyecto, tu explorador en VS Code debe verse de la siguiente manera:

```text
practica8-ml-scikit-learn/
│
├── .venv8/                      # Entorno virtual aislado
├── practica8a-0777.py           # Ejemplo 1: Regresión Lineal (Aprobar examen)
├── practica8b-0777.py           # Ejemplo 2: Clasificación (Aprobado / Reprobado)
├── practica8c-0777.py           # Ejemplo 3: Árbol de Decisión (Filtro Spam)
├── practica8d-0777.py           # Ejemplo 4: Agrupamiento K-Means (Grupos de clientes)
└── practica8e-0777.py           # Ejemplo 5: K-Vecinos más Cercanos (Deporte recomendado)

```

---

### **Paso 3: Códigos de Python explicados paso a paso**

1. **practica8a-0777.py:** Regresión Lineal Simple: Predicción de calificación según horas de estudio.
Un modelo de regresión lineal busca la "línea" que mejor predice un número continuo (en este caso, una calificación del 0 al 10).

```python
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento: [Horas de estudio]
# (Se usa doble corchete [[ ]] porque Scikit-learn espera una matriz 2D para la variable de entrada)
X_train = [[1], [2], [3], [5], [6], [8], [9]]

# Etiquetas (Respuestas conocidas): Calificación obtenida
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


2. **practica8b-0777.py:** Clasificación Logística: ¿Riesgo de reprobar la materia?.
A diferencia de la regresión, la clasificación asigna datos a **categorías de texto o etiquetas** (por ejemplo: "En Riesgo" o "A salvo").

```python
from sklearn.linear_model import LogisticRegression

# Datos: [Horas de estudio, Asistencias en porcentaje]
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


3. **practica8c-0777.py:** Árbol de Decisión: Clasificador de correos (Spam vs. No Spam).
Los árboles de decisión aprenden reglas tipo *Si / Entonces* de forma automática basadas en características.

```python
from sklearn.tree import DecisionTreeClassifier

# Características del correo: [Número de links, Contiene palabra 'Gratis' (1=Sí, 0=No)]
X_train = [
    [0, 0],  # Correo normal de un amigo
    [5, 1],  # "¡Ganaste un premio! Da clic aquí"
    [1, 0],  # Tarea del maestro con 1 enlace
    [7, 1],  # Oferta sospechosa
    [0, 1]   # Mensaje con la palabra gratis pero sin enlaces
]

# Etiquetas: "Normal" o "Spam"
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


4. **practica8d-0777.py:** Aprendizaje No Supervisado (K-Means): Agrupación de clientes por hábitos de compra.
K-Means agrupa datos automáticamente **sin conocer etiquetas previas**. Encuentra patrones o "tribus" por sí solo.

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

# 2. Ver a qué grupo (0 o 1) asignó a cada cliente
etiquetas_grupos = kmeans.labels_

print("--- SEGMENTACIÓN AUTOMÁTICA DE CLIENTES ---")
for i, cliente in enumerate(clientes):
    print(f"Cliente {i+1} {cliente} -> Grupo asignado: {etiquetas_grupos[i]}")

```


5. **practica8e-0777.py:** K-Vecinos más Cercanos (KNN): Recomendación de Deporte.
El algoritmo K-Nearest Neighbors (KNN) busca las observaciones pasadas más parecidas a la nueva para tomar una decisión.

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

# Deporte recomendado
y_train = ["Gimnasia", "Gimnasia", "Básquetbol", "Básquetbol", "Fútbol"]

# 1. Crear el clasificador KNN con K=3 (mira a sus 3 vecinos más cercanos)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# 2. Recomendar deporte a un alumno nuevo que mide 188 cm y pesa 88 kg
nuevo_alumno = [[188, 88]]
deporte_recomendado = modelo.predict(nuevo_alumno)

print("--- SISTEMA DE RECOMENDACIÓN DEPORTIVA ---")
print(f"Para una estatura de {nuevo_alumno[0][0]}cm y {nuevo_alumno[0][1]}kg, se recomienda: {deporte_recomendado[0]}")

```
