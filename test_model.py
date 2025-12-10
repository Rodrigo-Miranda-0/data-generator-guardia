"""
Test the trained model with real-world-like examples that weren't in training.
"""
from model.inference import PhishingDetector

# Load the trained model
detector = PhishingDetector("artifacts/phishing-model")

# Test cases - mix of obvious and tricky examples
test_emails = [
    # PHISHING (should be 1)
    {
        "text": """Estimado cliente, 
        
Hemos detectado actividad inusual en su cuenta bancaria. Para evitar el bloqueo 
permanente de su cuenta, debe verificar su identidad inmediatamente.

Haga clic aquí: http://banco-seguridad.net/verificar

Tiene 24 horas para completar este proceso.

Servicio al Cliente""",
        "expected": 1,
        "type": "Phishing - banco urgente"
    },
    {
        "text": """Tu contraseña de Microsoft 365 ha expirado. 
        
Actualiza tu contraseña ahora para continuar usando tu cuenta:
https://microsoft-password-reset.com/update

Si no actualizas en 48 horas, perderás acceso a tu correo.

Equipo de Microsoft""",
        "expected": 1,
        "type": "Phishing - password reset"
    },
    {
        "text": """URGENTE: Factura pendiente #INV-2024-001

Su factura de $2,500 USD está vencida. Para evitar acciones legales, 
realice el pago inmediatamente en el siguiente enlace:

www.pagos-empresa.com/factura/2024001

Departamento de Cobranzas""",
        "expected": 1,
        "type": "Phishing - factura falsa"
    },
    
    # LEGITIMATE (should be 0)
    {
        "text": """Hola equipo,

Les recuerdo que mañana tenemos la reunión semanal a las 10:00 AM 
en la sala de conferencias B.

Por favor confirmen asistencia.

Saludos,
María González
Gerente de Proyectos""",
        "expected": 0,
        "type": "Legítimo - reunión"
    },
    {
        "text": """Estimado Juan,

Adjunto encontrará el reporte trimestral de ventas que solicitó. 
Los números muestran un crecimiento del 15% respecto al trimestre anterior.

Quedo atento a sus comentarios.

Saludos cordiales,
Pedro Ramírez
Analista de Ventas""",
        "expected": 0,
        "type": "Legítimo - reporte"
    },
    {
        "text": """Buenas tardes,

Le confirmamos que su pedido #45678 ha sido enviado y llegará 
entre el 15 y 17 de diciembre.

Puede rastrear su envío en nuestra página web con su número de pedido.

Gracias por su compra.
Atención al Cliente - TiendaOnline""",
        "expected": 0,
        "type": "Legítimo - envío"
    },
    
    # TRICKY CASES (harder to classify)
    {
        "text": """Notificación: Su paquete está retenido en aduana.

Para liberar su envío, debe pagar los aranceles correspondientes ($45.00 USD).
Visite nuestra oficina en Av. Principal 123 o llame al 555-1234.

Servicio de Aduanas Nacional""",
        "expected": 1,  # Likely phishing but looks more legitimate
        "type": "Difícil - aduana (phishing)"
    },
    {
        "text": """IMPORTANTE: Actualización de datos requerida

Estimado empleado, como parte de nuestra auditoría anual, necesitamos 
que actualice sus datos en el portal de RRHH antes del viernes.

Ingrese a: https://rrhh.empresa.com/actualizar-datos

Recursos Humanos""",
        "expected": 0,  # Legitimate internal email
        "type": "Difícil - RRHH (legítimo)"
    },
]

print("=" * 70)
print("TESTING MODEL WITH REAL-WORLD-LIKE EXAMPLES")
print("=" * 70)

correct = 0
total = len(test_emails)

for i, test in enumerate(test_emails, 1):
    result = detector.predict_email(test["text"])
    predicted = 1 if result["label"] == "phishing" else 0
    confidence = result["score"]
    
    is_correct = predicted == test["expected"]
    correct += int(is_correct)
    
    status = "✅" if is_correct else "❌"
    pred_label = "PHISHING" if predicted == 1 else "LEGÍTIMO"
    exp_label = "PHISHING" if test["expected"] == 1 else "LEGÍTIMO"
    
    print(f"\n{status} Test {i}: {test['type']}")
    print(f"   Esperado: {exp_label} | Predicho: {pred_label} ({confidence:.1%})")

print("\n" + "=" * 70)
print(f"RESULTADO: {correct}/{total} correctos ({correct/total:.1%})")
print("=" * 70)

if correct == total:
    print("\n⚠️  100% en tests manuales también. El modelo puede estar aprendiendo")
    print("   patrones superficiales. Prueba con más ejemplos variados.")
elif correct / total < 0.7:
    print("\n⚠️  Rendimiento bajo en ejemplos reales. El modelo no generaliza bien.")
    print("   Necesitas datos de entrenamiento más diversos/reales.")
else:
    print("\n✅ Rendimiento razonable. El modelo parece generalizar decentemente.")

