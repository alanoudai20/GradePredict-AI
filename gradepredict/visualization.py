"""Four dataset-level charts for UCI Portuguese course grades."""
from .config import FEATURES
from .transformation import prepare_features

LABELS = {"studytime": "Weekly study-time category", "G1": "First-period grade /20", "G2": "Second-period grade /20"}

def create_figures(dataset):
    """Plot observed data without filling, clipping or removing zero grades."""
    import matplotlib.pyplot as plt
    features, final_grades = prepare_features(dataset)
    figures = {}
    try:
        for feature in FEATURES:
            figure, axes = plt.subplots(figsize=(8,5), constrained_layout=True)
            figures[f"{feature.lower()}_vs_final_grade"] = figure
            complete = features[feature].notna() & final_grades.notna()
            if feature == "studytime":
                # Each group contains every observed grade in that study category.
                groups = [final_grades[features[feature] == category].dropna().to_numpy()
                          for category in (1, 2, 3, 4)]
                axes.boxplot(groups, tick_labels=["Under 2 hours", "2–5 hours", "5–10 hours", "Over 10 hours"],
                             patch_artist=True,
                             boxprops={"facecolor": "#DCEEF5", "edgecolor": "#0F8B8D"},
                             medianprops={"color": "#172B4D", "linewidth": 2},
                             whiskerprops={"color": "#0F8B8D"}, capprops={"color": "#0F8B8D"},
                             flierprops={"marker": "o", "markersize": 4, "markeredgecolor": "#172B4D"})
                axes.set(xlabel="Weekly Study Time", ylabel="Final Course Grade (G3) /20",
                         title=f"Weekly Study Time vs Final Course Grade (n={int(complete.sum())})", ylim=(0,20))
            else:
                axes.scatter(features.loc[complete,feature], final_grades.loc[complete], alpha=0.3, s=18)
                axes.set(xlabel=LABELS[feature], ylabel="Final course grade G3 /20",
                         title=f"{LABELS[feature]} vs Final Course Grade (n={int(complete.sum())})", ylim=(-0.5,20.5))
            axes.grid(alpha=0.2)
        figure, axes = plt.subplots(figsize=(8,5), constrained_layout=True)
        figures['final_grade_distribution'] = figure
        axes.hist(final_grades.dropna(), bins=[value-0.5 for value in range(22)], color="#0F8B8D", edgecolor="white")
        axes.set(xlabel="Final course grade G3 /20", ylabel="Number of records",
                 title="Final Course Grade Distribution (zeros retained)", xlim=(-0.5,20.5))
        return figures
    except Exception:
        for figure in figures.values():
            plt.close(figure)
        raise
