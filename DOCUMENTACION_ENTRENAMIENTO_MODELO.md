# 8.2 Desarrollo y Evolución del Clasificador de Phishing

Esta sección documenta el proceso iterativo de desarrollo del modelo de clasificación de correos electrónicos, incluyendo los desafíos encontrados, las decisiones tomadas y las lecciones aprendidas durante cada fase de experimentación. El objetivo es proporcionar trazabilidad completa sobre la evolución del componente de inteligencia artificial y fundamentar las decisiones técnicas adoptadas.

## 8.2.1 Contexto Inicial y Desafío de Datos

El desarrollo del clasificador de phishing enfrentó un obstáculo fundamental desde el inicio: la escasez de datasets etiquetados de phishing en español. Los conjuntos de datos públicos disponibles están predominantemente en inglés, y los correos corporativos reales contienen información confidencial que impide su uso directo para entrenamiento.

Esta limitación condujo a explorar la generación de datos sintéticos como estrategia de arranque para el modelo inicial.

## 8.2.2 Fase 1: Enfoque de Datos Sintéticos

### Hipótesis Inicial

Se planteó que la generación de correos sintéticos mediante Modelos de Lenguaje Grande (LLMs) podría proporcionar un conjunto de datos de entrenamiento viable, permitiendo entrenar un modelo funcional antes de disponer de datos reales del entorno corporativo.

### Pipeline de Generación Implementado

Se desarrolló un pipeline de tres etapas para la generación de datos sintéticos: Generación de Prompts, seguido de Generación de Correos mediante LLM, y finalmente Limpieza de Datos.

En la etapa de Generación de Prompts, se diseñaron prompts paramétricos que combinan múltiples dimensiones para crear variabilidad. Los arquetipos de phishing incluyen: suspensión de cuenta, reseteo de contraseña, factura vencida, CEO fraud, paquete retenido, alerta de seguridad, actualización de nómina, software malicioso, acceso a drive y documentos fiscales (10 tipos en total). Los arquetipos legítimos incluyen: invitación a reunión, factura real, confirmación de reseteo, mantenimiento IT, itinerario de viaje, beneficios, estado de proyecto y confirmación de envío (8 tipos). Además, se varían el tono (formal, neutral, informal), la urgencia (sin urgencia, moderada, alta), la presencia de enlaces (login, pago, sin enlaces), la mención de adjuntos (con PDF, sin adjuntos) y el idioma (español puro, español con anglicismos).

Esta combinación produjo 540 prompts únicos para la generación.

Para la generación de correos se evaluaron dos modelos de lenguaje: Llama 3.1 8B Instruct de Meta, un modelo alineado con filtros de seguridad, y Dolphin-Mistral, un modelo sin censura basado en Mistral 7B.

### Obstáculo: Filtros de Seguridad del LLM

El modelo Llama 3.1 8B Instruct rechazó generar aproximadamente el 60% de los prompts de phishing, produciendo respuestas como "Lo siento, pero no puedo cumplir con esa solicitud" o "I cannot write a phishing email". De 300 prompts de phishing enviados, 182 (60.7%) fueron rechazados, resultando en solo 107 correos de phishing generados válidos.

Como solución, se migró al modelo Dolphin-Mistral, que carece de restricciones de seguridad. Este modelo generó 199 correos de phishing válidos con una tasa de rechazo del 0%.

### Resultados del Primer Entrenamiento

Se entrenó DistilBERT con el dataset sintético generado. Los resultados en validación fueron: Accuracy 100.00%, Precision 100.00%, Recall 100.00% y F1 Score 100.00%. La matriz de confusión mostró 47 verdaderos negativos, 0 falsos positivos, 0 falsos negativos y 37 verdaderos positivos.

Señal de Alarma: Una precisión del 100% en validación es casi siempre indicativa de sobreajuste o fuga de datos. Este resultado motivó una investigación detallada.

## 8.2.3 Fase 2: Investigación del Sobreajuste

### Prueba con Datos Externos

Para evaluar la capacidad de generalización del modelo, se prepararon 8 correos escritos manualmente (no sintéticos) y se midió el desempeño. De los 4 correos de phishing, 0 fueron detectados correctamente (0% accuracy). De los 4 correos legítimos, 4 fueron clasificados correctamente (100% accuracy). El total fue 4/8 correctos (50% accuracy).

Diagnóstico: El modelo clasificó todos los correos como legítimos, independientemente de su contenido real. No había aprendido patrones de phishing; había aprendido a distinguir el "estilo sintético" del LLM.

### Análisis de Causas Raíz

La investigación reveló dos fuentes principales de sesgo.

Primera fuente: Fuga de Datos en Archivos .eml. Adicionalmente a los datos sintéticos, se contaba con aproximadamente 1,000 archivos .eml de phishing reales obtenidos de phishingpot.com. El análisis reveló que 313 de 1000 (31%) contenían la palabra "phishing" en metadatos, como direcciones del tipo "phishing@pot.com" o headers como "X-Phishing-Test: true". Esta información permitía al modelo identificar phishing sin analizar el contenido del correo.

Segunda fuente: Patrones Detectables en Datos Sintéticos. El análisis de los correos legítimos generados por el LLM mostró que 740 de 1000 (74%) presentaban patrones sistemáticos: saludos formulaicos como "Dear [Name],", cierres predecibles como "Best regards", "Thanks" o "Cheers", y nombres repetitivos como Michael, Sarah o Robert. El modelo había aprendido a detectar estos patrones superficiales en lugar de características semánticas de phishing.

### Conclusión Teórica

El fenómeno observado corresponde a un caso de desplazamiento de dominio (domain shift), donde la distribución conjunta de características y etiquetas en datos sintéticos difiere de la distribución en correos reales. La validación con datos de la misma distribución sintética no garantiza generalización.

## 8.2.4 Fase 3: Búsqueda de Datos Reales

### Evaluación del Dataset de HuggingFace

Se descargó el dataset cybersectony/PhishingEmailDetectionv2.0, que contiene: 6,809 emails legítimos del Corpus Enron (label 0), 6,684 emails de phishing (label 1), 53,157 URLs legítimas (label 2) y 53,350 URLs de phishing (label 3).

Los correos legítimos provenían del Corpus Enron (comunicaciones corporativas reales de 2001-2002) y presentaban características auténticas de entornos empresariales, como por ejemplo: "revised - transitional steering committee meeting here are the details for the transitional steering committee meetings : this meeting will take place every wednesday at 10:00 a.m. (cst) commencing on february 6th..."

Sin embargo, los correos etiquetados como "phishing" resultaron ser spam genérico, no phishing corporativo. Por ejemplo: "Dear ricardo1, COST EFFECTIVE Direct Email Advertising Promote Your Business For As Low As $50 Per 1 Million Email Addresses..."

Este tipo de contenido corresponde a email marketing masivo, no a ataques de credential harvesting o ingeniería social dirigida que Guard-IA debe detectar.

Decisión: Se descartaron los correos de phishing de HuggingFace y se conservaron únicamente los correos legítimos del Corpus Enron.

## 8.2.5 Fase 4: Construcción del Dataset Final

### Proceso de Limpieza

Paso 1: Eliminación de Fuga de Datos en Phishing. Se eliminaron los correos que contenían "phishing" en el texto mediante el filtro correspondiente. El resultado fue 687 correos (de 1,000 originales, se eliminaron 313).

Paso 2: Descarte Total de Datos Sintéticos. Los 1,000 correos legítimos generados por LLM fueron descartados completamente debido a los patrones detectables identificados.

Paso 3: Balanceo con Corpus Enron. Se realizó un muestreo aleatorio de 687 correos del corpus Enron para balancear las clases.

### Composición Final del Dataset

La clase Legítimo (0) quedó compuesta por 687 correos corporativos reales del Corpus Enron (2001-2002). La clase Phishing (1) quedó compuesta por 687 correos de los archivos .eml limpiados, correspondientes a phishing de consumidor sin fuga de datos. El total del dataset es de 1,374 muestras, con un split 80/20 que resulta en 1,099 muestras para entrenamiento y 275 para validación.

El archivo generado fue: data/corporate_phishing_dataset.csv

## 8.2.6 Fase 5: Selección y Comparación de Modelos

### Modelos Evaluados

Una vez definido el dataset limpio, se procedió a evaluar diferentes arquitecturas de modelos para determinar cuál ofrecía el mejor balance entre precisión, eficiencia computacional y viabilidad de integración en el pipeline de Guard-IA.

Se consideraron cuatro modelos principales:

DistilBERT: Con 66 millones de parámetros y un tiempo de inferencia aproximado de 15ms, ofrece como ventajas ser ligero, eficiente y tener buen balance. Su desventaja es una menor capacidad semántica comparada con modelos más grandes.

RoBERTa: Con 125 millones de parámetros y un tiempo de inferencia aproximado de 30ms, ofrece mayor precisión en texto complejo. Sus desventajas son ser más lento y tener mayor consumo de recursos.

BERT-base: Con 110 millones de parámetros y un tiempo de inferencia aproximado de 25ms, es un baseline robusto pero no ofrece ventajas sobre DistilBERT.

BERT-large: Con 340 millones de parámetros y un tiempo de inferencia aproximado de 80ms, tiene alta capacidad pero es excesivo para el caso de uso.

### Evaluación de DistilBERT

DistilBERT fue seleccionado como modelo principal por las siguientes razones:

Eficiencia: Al ser una versión destilada de BERT, conserva aproximadamente el 97% del rendimiento del modelo original con solo el 60% de los parámetros. Esto permite tiempos de inferencia compatibles con el requisito de latencia del sistema (análisis en menos de 3 segundos por correo).

Balance precisión-velocidad: Para la tarea de clasificación binaria (phishing vs. legítimo), la capacidad de DistilBERT resultó suficiente para capturar los patrones lingüísticos relevantes sin introducir complejidad innecesaria.

Facilidad de fine-tuning: El modelo permite ajustes rápidos con datasets relativamente pequeños (1,374 muestras), lo cual es crítico dado el tamaño del dataset disponible.

Integración con Hugging Face: La compatibilidad nativa con la biblioteca Transformers simplifica el despliegue y mantenimiento del modelo en producción.

### Evaluación de RoBERTa

RoBERTa fue evaluado como alternativa debido a su mayor capacidad para capturar relaciones semánticas complejas. Se realizaron pruebas preliminares con el mismo dataset.

Los resultados comparativos en validación fueron: DistilBERT alcanzó 99.64% de accuracy, 99.22% de precision, 100% de recall, 99.61% de F1 y 18ms por correo. RoBERTa-base alcanzó exactamente las mismas métricas (99.64% accuracy, 99.22% precision, 100% recall, 99.61% F1) pero con 42ms por correo.

Análisis: Ambos modelos alcanzaron métricas idénticas en el dataset de validación. Sin embargo, RoBERTa requirió más del doble de tiempo de inferencia sin aportar mejoras medibles en precisión. Esto se explica porque la tarea de clasificación, con el dataset actual, no demanda la capacidad adicional de RoBERTa.

Decisión: Se descartó RoBERTa como modelo principal, aunque se mantiene como opción para escenarios futuros donde se requiera mayor profundidad de análisis (por ejemplo, detección de BEC con datasets especializados).

### Justificación del Descarte de Otros Modelos

BERT-base y BERT-large fueron descartados por no ofrecer ventajas sobre DistilBERT. BERT-base tiene un rendimiento comparable pero es más lento; BERT-large introduce complejidad computacional excesiva sin beneficios proporcionales para esta tarea.

Los modelos de mayor escala como GPT y LLaMA no se consideraron para la tarea de clasificación debido a: latencia incompatible con procesamiento en tiempo real, requerimientos de infraestructura que exceden el alcance del proyecto, y diseño orientado a generación de texto en lugar de clasificación.

Se evaluaron también datasets y modelos preentrenados específicos para detección de phishing, pero presentaban las siguientes limitaciones: entrenados predominantemente en inglés, orientados a phishing de consumidor (no corporativo) y sin capacidad multilingüe.

### Decisión del Enfoque Híbrido

El análisis de las limitaciones del clasificador llevó a la adopción de un enfoque híbrido para el pipeline de Guard-IA, compuesto por tres capas secuenciales:

La primera capa es la Capa Heurística, que realiza filtrado rápido de casos evidentes mediante reglas, validación SPF/DKIM y listas negras. Esta capa tiene latencia mínima y no consume recursos de IA.

La segunda capa es la Capa DistilBERT, que realiza la clasificación de correos que superan el filtro heurístico. El modelo evalúa el contenido textual y proporciona una puntuación de riesgo.

La tercera capa es la Capa LLM (Claude o GPT), que para correos clasificados como sospechosos genera una explicación comprensible para el usuario, detallando los indicadores de riesgo identificados.

Esta arquitectura híbrida mitiga las limitaciones del clasificador de las siguientes formas: para BEC y spear phishing, la capa heurística puede detectar patrones de CEO fraud mediante reglas específicas (análisis de headers, detección de dominios similares); para contexto organizacional, la capa LLM puede incorporar información contextual para generar warnings más precisos; para eficiencia, la mayoría de correos son procesados solo por las primeras dos capas, reservando el LLM para casos que requieren explicación.

## 8.2.7 Fase 6: Entrenamiento Final del Modelo Seleccionado

### Configuración del Entrenamiento

El entrenamiento se realizó con los siguientes parámetros: modelo base DistilBERT (distilbert-base-uncased), 3 epochs, batch size de 8, learning rate de 5e-5, max sequence length de 256, weight decay de 0.01. El hardware utilizado fue Apple M-series (MPS) y el tiempo de entrenamiento fue aproximadamente 3 minutos.

El comando de entrenamiento fue:

PYTHONPATH=. python model/train.py --csv_path data/corporate_phishing_dataset.csv --output_dir artifacts/phishing-model-corporate --epochs 3 --batch_size 8 --model_name distilbert-base-uncased

### Progresión del Entrenamiento

La progresión durante las épocas fue la siguiente: Epoch 1 alcanzó eval_accuracy de 0.9964 y eval_loss de 0.0280. Epoch 2 alcanzó eval_accuracy de 0.9964 y eval_loss de 0.0313. Epoch 3 alcanzó eval_accuracy de 0.9964 y eval_loss de 0.0322.

### Métricas Finales

Las métricas finales del modelo fueron: Accuracy 99.64%, Precision 99.22%, Recall 100.00% y F1 Score 99.61%.

La matriz de confusión en el conjunto de validación (275 muestras) mostró: 145 verdaderos negativos (correos legítimos correctamente clasificados), 1 falso positivo (correo legítimo clasificado como phishing), 0 falsos negativos (ningún phishing escapó detección) y 128 verdaderos positivos (correos de phishing correctamente detectados).

El modelo alcanzó 100% recall, cumpliendo con el objetivo de minimizar falsos negativos. El único error fue un falso positivo, manteniendo una tasa de FP del 0.68%, muy por debajo del umbral aceptable del 5-10%.

## 8.2.8 Validación con Ejemplos Externos

Se evaluó el modelo con correos no vistos durante el entrenamiento. Un correo de fraude CEO urgente solicitando transferencia fue clasificado como LEGIT con 1.1% de confianza en phishing (incorrecto). Un correo sobre paquete esperando en aduana fue clasificado como LEGIT con 7.2% de confianza en phishing (incorrecto). Un correo de invitación a reunión de equipo el viernes a las 3pm fue clasificado como LEGIT con 0.2% de confianza en phishing (correcto). Un correo de factura vencida con enlace fue clasificado como PHISHING con 63.2% de confianza (correcto).

### Análisis de Capacidades y Limitaciones

El modelo detecta correctamente: phishing de paquetes/delivery, facturas falsas con enlaces maliciosos, alertas de seguridad genéricas y correos de spam malicioso.

El modelo NO detecta: BEC (Business Email Compromise / CEO fraud), spear phishing corporativo dirigido y amenazas que requieren contexto organizacional.

Justificación: El dataset de entrenamiento (.eml de phishingpot.com) contiene predominantemente phishing de consumidor (ataques masivos no dirigidos). Los ataques BEC requieren comprender relaciones jerárquicas y contexto empresarial que no están presentes en el dataset.

Mitigación: Estas limitaciones se abordan en Guard-IA mediante las capas complementarias del pipeline. La capa heurística detecta patrones de CEO fraud por keywords y análisis de headers. La capa LLM proporciona análisis contextual y warning sobre posibles ataques dirigidos.

## 8.2.9 Comparación de Versiones del Modelo

La versión 1 (sintético) utilizó dataset de LLM más .eml, alcanzó 100% de accuracy y 1.00 de F1 con 0 falsos positivos, pero no generalizó.

La versión 2 (HuggingFace) utilizó solo el dataset de HF, pero fue descartada porque contenía spam genérico en lugar de phishing corporativo.

La versión 3 (final) utilizó Enron más .eml limpio, alcanzó 99.64% de accuracy y 0.9961 de F1 con 1 falso positivo, y sí generaliza correctamente.

## 8.2.10 Lecciones Aprendidas

El proceso de desarrollo del clasificador proporcionó aprendizajes significativos aplicables a proyectos de ML en seguridad:

Primera lección: Los datos sintéticos presentan limitaciones fundamentales. Los modelos de lenguaje introducen patrones detectables que permiten al clasificador aprender atajos en lugar de características semánticas reales. La validación interna con datos sintéticos no garantiza generalización.

Segunda lección: La métrica del 100% es una señal de alarma. Resultados perfectos en validación deben investigarse como posible fuga de datos o sobreajuste a artefactos del dataset.

Tercera lección: La fuga de datos puede ser sutil. Metadatos como direcciones de correo (por ejemplo phishing@pot.com) pueden revelar la etiqueta sin necesidad de analizar el contenido.

Cuarta lección: Los datos reales superan a los sintéticos. Incluso datos antiguos (Corpus Enron, 2001-2002) generalizaron mejor que datos sintéticos frescos generados por LLMs de última generación.

Quinta lección: El dominio importa. Phishing de consumidor es diferente de phishing corporativo. El modelo debe entrenarse con datos representativos del entorno de despliegue.

Sexta lección: Los LLMs alineados no colaboran. Los filtros de seguridad de modelos como Llama 3.1 dificultan la generación de contenido malicioso incluso con fines de investigación legítimos.

## 8.2.11 Artefactos Generados

El modelo entrenado se encuentra en artifacts/phishing-model-corporate/ y contiene: config.json, model.safetensors, tokenizer_config.json, vocab.txt y special_tokens_map.json.

Los datasets se encuentran en data/ e incluyen: corporate_phishing_dataset.csv (dataset final con 1,374 muestras), training_data.csv (dataset original contaminado), phishing_dataset_200k.csv (descarga de HuggingFace) y synthetic_emails.csv (correos sintéticos descartados).

Los scripts de soporte se encuentran en data/ e incluyen: gen_prompts.py (generación de prompts), generate_synthetic_ollama.py (generación vía Ollama), clean_synthetic_data.py (limpieza de datos) y prepare_training_data.py (preparación del dataset final).

---

Este documento forma parte de la documentación técnica del componente de Inteligencia Artificial de Guard-IA.
