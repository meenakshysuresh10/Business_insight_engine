import shap
import matplotlib.pyplot as plt

def explain(model, df, sample_index=0):
    X = df[['value']].copy()

    # Use the model's decision_function as a callable
    explainer = shap.Explainer(model.decision_function, X)

    shap_values = explainer(X)

    # Show explanation for one sample
    shap.plots.waterfall(shap_values[sample_index], show=False)
    plt.title("SHAP Explanation for Anomaly")
    plt.tight_layout()
    plt.show()
