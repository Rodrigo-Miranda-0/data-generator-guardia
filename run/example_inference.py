"""
Example script for running inference and explanations.
"""

from model.inference import PhishingDetector
from model.explain import AttentionExplainer


def main() -> None:
    model_dir = "artifacts/phishing-model"
    detector = PhishingDetector(model_dir)
    explainer = AttentionExplainer(model_dir)

    samples = [
        "Estimado usuario, su cuenta necesita verificación urgente. Haga clic en el enlace.",
        "Recordatorio de la reunión mensual mañana a las 10 AM en la sala principal.",
    ]

    print("Predictions:")
    for text in samples:
        print(detector.predict_email(text))

    print("\nBatch prediction:")
    print(detector.predict_batch(samples))

    print("\nExplanation for first sample:")
    print(explainer.explain_email(samples[0], top_k=5))


if __name__ == "__main__":
    main()




