import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use('ggplot')
    df = pd.DataFrame({
        "Model": ["DecisionTree", "AdaBoost", "RandomForest", "GradientBoostedClassifier"],
        "Accuracy_CV": [0.77325, 0.792375, 0.7286, 0.796625],
        "Precision_CV": [0.814481, 0.806253, 0.7824419, 0.8132654],
        "Recall_CV": [0.827765, 0.874798, 0.785772, 0.8715670],
        "Accuracy_HO": [0.7420, 0.7787, 0.7319, 0.7831],
        "Precision_HO": [0.779356, 0.79869, 0.785396, 0.80606],
        "Recall_HO": [0.82501, 0.861913, 0.792710, 0.85822]
    })
    df = df.sort_values(by="Accuracy_CV", ascending=False)
    print(df.to_markdown())

    df_ho = df.drop(columns=['Accuracy_CV', 'Precision_CV', 'Recall_CV'])
    df_cv = df.drop(columns=['Accuracy_HO', 'Precision_HO', 'Recall_HO'])

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    
    ax.bar(df_cv['Model'], df_cv['Accuracy_CV'], color='b', label="Cross Validation")
    ax.bar(df_ho['Model'], df_ho['Accuracy_HO'], color='skyblue', label="Holdout", width=0.5)
    ax.set_ylim((0.7, 0.9)) 
    ax.legend()
    plt.savefig("images/score_plot.png")