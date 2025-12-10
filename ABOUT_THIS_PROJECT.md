# 📧 Generación de Datos Sintéticos

Este documento explica cómo funciona el pipeline de generación de correos sintéticos.

---

## 📁 Estructura de archivos

```
data/
├── gen_prompts.py              # Genera los prompts (NO usa LLM)
├── generate_synthetic_ollama.py # Genera correos usando Ollama
├── generate_synthetic_kobold.py # Genera correos usando KoboldCpp
└── generate_synthetic.py        # Genera correos usando HuggingFace

prompts.txt                      # Los 540 prompts generados
synthetic_emails.csv             # Output: correos generados con labels
```

---

## 🔄 Pipeline completo

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ gen_prompts.py  │ ──▶ │   prompts.txt   │ ──▶ │ generate_*.py   │
│ (combinaciones) │     │ (540 prompts)   │     │ (llama 3.1)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ synthetic_emails│
                                               │     .csv        │
                                               └─────────────────┘
```

---

## Paso 1: Generar prompts (`gen_prompts.py`)

Este script **NO llama a ningún LLM**. Solo genera combinaciones de parámetros para crear prompts variados.

### Arquetipos definidos

**Phishing (10 tipos):**
- Suspensión urgente de cuenta
- Reseteo de contraseña
- Factura vencida
- Transferencia CEO fraud
- Paquete retenido en aduana
- Alerta de seguridad (login inusual)
- Actualización de nómina/beneficios
- Actualización de software (adjunto)
- Acceso a drive compartido
- Descarga de documentos fiscales

**Legítimos (8 tipos):**
- Invitación a reunión
- Notificación de factura real
- Confirmación de reseteo de contraseña
- Aviso de mantenimiento IT
- Actualización de itinerario de viaje
- Recordatorio de inscripción a beneficios
- Actualización de estado de proyecto
- Confirmación de envío

### Parámetros que se combinan

| Parámetro | Opciones |
|-----------|----------|
| Tono | `formal`, `neutral`, `slightly informal` |
| Urgencia | `no urgency`, `mild urgency`, `high urgency` |
| Links | `login link`, `payment link`, `no links` |
| Adjuntos | `PDF attachment`, `no attachments` |
| Idioma | `pure Spanish`, `Spanish with English words` |
| Longitud | `under 120 words`, `under 180 words` |

### Ejemplo de prompt generado

```
Write a phishing Spanish corporate email about urgent account suspension. 
Tone: formal. Urgency: high urgency. include a login link. mention a PDF attachment. 
Language: pure Spanish. Length: under 120 words. Avoid placeholders; make it realistic.
```

### Comando

```bash
python data/gen_prompts.py --output prompts.txt --samples_per_prompt 30
```

Esto genera **540 prompts** (10 phishing + 8 legit = 18 arquetipos × 30 variantes).

---

## Paso 2: Generar correos con el LLM

Tenemos **3 scripts** según qué herramienta uses para correr el modelo:

### Opción A: Ollama (recomendado) ✅

```bash
# Primero asegúrate que Ollama esté corriendo con el modelo
ollama run llama3.1:8b

# En otra terminal, ejecuta el script
python data/generate_synthetic_ollama.py \
    --prompts_file prompts.txt \
    --output_csv synthetic_emails.csv \
    --model llama3.1:8b \
    --temperature 0.9 \
    --num_samples_per_prompt 1
```

**Características:**
- Se conecta a `http://localhost:11434`
- Guarda progreso incrementalmente (si se interrumpe, continúa donde quedó)
- Usa `--no_resume` para empezar de cero

### Opción B: KoboldCpp

```bash
# Primero inicia KoboldCpp
koboldcpp.exe --model llama-3-8b-instruct.Q4_K_M.gguf --port 5001 --api

# Luego ejecuta
python data/generate_synthetic_kobold.py \
    --prompts_file prompts.txt \
    --output_csv synthetic_emails.csv \
    --api_url http://localhost:5001/api/v1/generate
```

### Opción C: HuggingFace Transformers

```bash
python data/generate_synthetic.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --prompts_file prompts.txt \
    --output_csv synthetic_emails.csv
```

**Nota:** Requiere acceso aprobado en HuggingFace y ~16GB de VRAM.

---

## 📊 Output: `synthetic_emails.csv`

El CSV tiene 2 columnas:

| Columna | Descripción |
|---------|-------------|
| `email_text` | El correo generado completo |
| `label` | `1` = phishing, `0` = legítimo |

### Cómo se asigna el label

El label se infiere automáticamente del prompt:
- Si el prompt contiene `"phishing"` → `label = 1`
- Si el prompt contiene `"legitimate"` → `label = 0`

```python
def infer_label_from_prompt(prompt: str) -> int:
    if "phishing" in prompt.lower():
        return 1
    if "legitimate" in prompt.lower():
        return 0
    return 1  # default
```

---

## 📝 Ejemplo de correo generado

**Prompt:**
```
Write a phishing Spanish corporate email about security alert unusual login...
```

**Output (label=1):**
```
Nota importante

Buenos días,

Necesitamos su atención inmediata sobre el estado de su cuenta. 
Según nuestras políticas de seguridad, hemos detectado transacciones 
sospechosas que requieren investigación inmediata.

Por razones de seguridad, estamos obligados a suspender su acceso 
si no responde: confirme su identidad con el número de cuenta 
asociada a su nombre.

El acceso a su cuenta se suspenderá en breve si no tenemos 
confirmación.

Por favor, responda con el número de cuenta y su nombre completo.

Equipo de Seguridad
```

---

## ⚠️ Notas importantes

1. **El modelo a veces se niega** a generar phishing (safety filters). Verán líneas como:
   ```
   "Lo siento, pero no puedo cumplir con esa solicitud.",1
   ```
   Estos se pueden filtrar después.

2. **Tiempo estimado:** ~1-2 segundos por correo. Con 540 prompts ≈ 15-30 minutos.

3. **Para generar más variedad:** Usa `--num_samples_per_prompt 5` para generar 5 correos por prompt (2700 total).

---

## 🚀 Comandos rápidos

```bash
# Generar prompts frescos
python data/gen_prompts.py --output prompts.txt --samples_per_prompt 30

# Generar correos con Ollama
python data/generate_synthetic_ollama.py

# Generar múltiples muestras por prompt
python data/generate_synthetic_ollama.py --num_samples_per_prompt 5 --output_csv big_dataset.csv
```
