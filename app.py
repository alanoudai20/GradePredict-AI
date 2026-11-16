"""GradePredict AI Streamlit interface. Run only when dependencies are permitted."""

import hashlib
import logging

try:
    import streamlit as st
except Exception as error:
    raise SystemExit(
        "GradePredict AI could not start Streamlit. Check the project interpreter "
        "and ask your administrator to review any Windows Application Control block."
    ) from error

from gradepredict.cleaning import clean_dataset
from gradepredict.config import DEFAULT_INPUT_PATH, DEFAULT_MODEL_DIR, FEATURES, PREDICTION_BOUNDS, RANDOM_STATE, STUDY_TIME_LABELS
from gradepredict.loading import load_dataset
from gradepredict.validation import validate_dataset

LOGGER = logging.getLogger(__name__)
MODEL_PATH = DEFAULT_MODEL_DIR / "linear_regression.joblib"
PAGES = ("Home", "Prediction", "Data Analysis")


def file_fingerprint(path):
    """Content-based cache keys change when the dataset or saved model changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@st.cache_data(show_spinner=False)
def get_project_data(dataset_fingerprint):
    """Read and clean once per dataset version; never write back to the CSV."""
    original = load_dataset(DEFAULT_INPUT_PATH)
    report = validate_dataset(original)
    report.require_schema()
    cleaned = clean_dataset(original)
    if not cleaned.dataset.rows:
        raise ValueError("No usable records remain after cleaning.")
    return original, report, cleaned


@st.cache_resource(show_spinner=False)
def get_verified_model(dataset_fingerprint, model_fingerprint):
    """Load a traceable trained model or train once, then evaluate real predictions.

    Cache keys include data/model contents. Cached pipelines are only used for
    prediction after construction; user input never refits the shared pipeline.
    """
    from sklearn.linear_model import LinearRegression
    from gradepredict.modeling import (build_model_pipeline, evaluate_predictions,
                                       generate_predictions, load_model,
                                       prepare_training_data, train_model)

    _, _, cleaned = get_project_data(dataset_fingerprint)
    training_features, testing_features, training_scores, testing_scores = prepare_training_data(cleaned.dataset)
    if model_fingerprint != "missing":
        model = load_model(MODEL_PATH)
        # A fitted object alone does not prove its evaluation data was held out.
        if getattr(model, 'gradepredict_source_sha256_', None) != dataset_fingerprint:
            raise ValueError("Saved model has no matching dataset provenance.")
        if getattr(model, 'gradepredict_training_records_', None) != list(training_features.index):
            raise ValueError("Saved model training records do not match the expected split.")
        if not isinstance(model.named_steps['regressor'], LinearRegression):
            raise ValueError("The saved model must be the project's Linear Regression model.")
        model_source = "Saved Linear Regression model"
    else:
        model = train_model(build_model_pipeline(), training_features, training_scores)
        model_source = "Linear Regression trained for this dataset"

    predictions = generate_predictions(model, testing_features)
    metrics = evaluate_predictions(testing_scores, predictions)
    baseline = train_model(build_model_pipeline('baseline'), training_features, training_scores)
    baseline_metrics = evaluate_predictions(testing_scores, generate_predictions(baseline, testing_features))
    return model, metrics, baseline_metrics, model_source, len(training_features), len(testing_features)


def show_error(message):
    """Keep technical tracebacks in server logs, not in the student interface."""
    LOGGER.exception(message)
    st.error(message)
    st.info("Check the original dataset and project dependencies. If Windows blocks a library, "
            "ask your administrator to review it before trying again. This action did not complete.")


def apply_dashboard_style():
    """Small presentation-only stylesheet; widgets stay native Streamlit."""
    st.markdown("""
    <style>
    .stMainBlockContainer {max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-size: 2rem !important; letter-spacing: -0.04em;}
    h2 {font-size: 1.35rem !important;}
    h3 {font-size: 1.1rem !important;}
    .st-key-hero {background: #172B4D; border-radius: 18px; padding: 1.7rem;}
    .st-key-hero h1, .st-key-hero p {color: #FFFFFF;}
    .st-key-hero h1 {max-width: 700px; font-size: clamp(1.8rem, 3vw, 2.6rem) !important;}
    .st-key-result {background: #FFFFFF; border-top: 4px solid #0F8B8D; border-radius: 14px;}
    .result-score {font-size: 3.4rem; font-weight: 750; color: #0F8B8D; line-height: 1.2;}
    .result-label {color: #172B4D; font-weight: 600; margin-bottom: 0.5rem;}
    .st-key-workflow, .st-key-project-facts, .st-key-analysis-facts {background: #FFFFFF; border-radius: 14px;}
    [data-testid="stSidebar"] {border-right: 1px solid #DCE4ED;}
    @media (max-width: 640px) {
        .stMainBlockContainer {padding-top: 1rem;}
        .st-key-hero {padding: 1rem;}
    }
    </style>
    """, unsafe_allow_html=True)


def show_dataset_limitations(report):
    with st.expander("Dataset Limitations"):
        st.markdown("- **Prediction Timing:** Predict after first-period (G1) and second-period (G2) grades are available.\n"
                    "- **Dataset Scope:** 649 records from two Portuguese schools.\n"
                    "- **Data Limitations:** All 15 zero final grades are retained because their individual causes are unknown.\n"
                    "- **Model Limitations:** Predictions are estimates and may not generalize to students from other schools or countries.")
        st.markdown("Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance). Cortez, P. (2008), DOI: 10.24432/C5TG7T. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source values unchanged; a feature subset is used.")


def open_prediction_page():
    st.session_state['page'] = "Prediction"


def show_home():
    with st.container(key="hero", border=True):
        st.caption("GRADEPREDICT AI  /  ACADEMIC INSIGHTS")
        st.title("Predict Your Academic Performance")
        st.write("GradePredict AI turns weekly study category and first- and second-period grades into an "
                 "informed final-course-grade estimate /20 using Machine Learning.")
        st.caption("For the Portuguese course: predict after G2 is available and before the final grade G3 is known.")
        st.button("Start Prediction", type="primary", on_click=open_prediction_page)
    st.write("")
    try:
        original, report, _ = get_project_data(file_fingerprint(DEFAULT_INPUT_PATH))
    except Exception:
        show_error("The project dataset is unavailable.")
        return
    with st.container(key="project-facts", border=True):
        records, features, model = st.columns(3)
        records.metric("Dataset records", f"{len(original.rows):,}")
        features.metric("Prediction features", len(FEATURES))
        model.caption("Model")
        model.subheader("Linear Regression")
    st.subheader("From your information to an informed estimate")
    steps = (
        ("01", "Enter Your Information", "Select weekly study time and enter your G1 and G2 grades."),
        ("02", "AI Analyzes Your Data", "A trained regression model uses these three measurements."),
        ("03", "Get Your Prediction", "View your estimate and understand the model's measured performance."),
    )
    with st.container(key="workflow"):
        for column, (number, title, description) in zip(st.columns(3), steps):
            with column:
                with st.container(border=True, height="stretch"):
                    st.caption(f"STEP {number}")
                    st.markdown(f"**{title}**")
                    st.write(description)
    st.caption("Final Course Grade Prediction Using Machine Learning · ARTI303 university project. "
               "Estimates support understanding; they do not guarantee academic results.")
    show_dataset_limitations(report)


def show_prediction():
    st.caption("ACADEMIC INSIGHTS / PREDICTION")
    st.title("Your final-course-grade estimate /20")
    st.write("Estimate the final course grade after G2 is available. Three inputs, one explainable model.")
    form_column, result_column = st.columns([1.1, 1], gap="large")
    with form_column:
        with st.form("prediction_form"):
            st.subheader("Your information")
            studytime = st.selectbox("Weekly Study Time", options=list(STUDY_TIME_LABELS), index=1,
                                      format_func=lambda value: STUDY_TIME_LABELS[value], key="studytime")
            first_grade = st.number_input("First-period Grade (G1)", min_value=0.0, max_value=20.0,
                                           value=12.0, step=1.0, format="%.1f", key="G1")
            second_grade = st.number_input("Second-period Grade (G2)", min_value=0.0, max_value=20.0,
                                            value=13.0, step=1.0, format="%.1f", key="G2")
            inputs = (studytime, first_grade, second_grade)
            submitted = st.form_submit_button("Predict Final Course Grade", type="primary", width="stretch")
            st.caption("Select category 1-4; grades must be 0-20. Predict only after G2 is available.")
        with st.expander("Number formatting"):
            st.caption("Numeric controls may follow your browser or system language. Decimal formatting is "
                       "consistent, but the browser may localize digit shapes. No locale settings are changed.")
            browser_locale = getattr(st.context, 'locale', None)
            if browser_locale:
                st.write(f"Browser locale: {browser_locale}")
    with result_column:
        with st.container(key="result", border=True):
            st.markdown('<div class="result-label">Predicted Final Course Grade /20</div>', unsafe_allow_html=True)
            result_content = st.empty()
            if not submitted:
                with result_content.container():
                    st.subheader("Ready when you are")
                    st.write("Complete the form and select Predict Final Course Grade to see your model-generated estimate.")
                    st.caption("No score is shown until a prediction succeeds.")
    if not submitted:
        try:
            _, report, _ = get_project_data(file_fingerprint(DEFAULT_INPUT_PATH))
            show_dataset_limitations(report)
        except Exception:
            show_error("Dataset information is unavailable.")
        return

    from gradepredict.prediction import predict_final_grade, validate_prediction_inputs
    try:
        validate_prediction_inputs(*inputs)
    except ValueError as error:
        st.error(str(error))
        return
    try:
        with st.spinner("Preparing the model and checking evaluation results…"):
            dataset_key = file_fingerprint(DEFAULT_INPUT_PATH)
            _, report, _ = get_project_data(dataset_key)
            model_key = file_fingerprint(MODEL_PATH) if MODEL_PATH.is_file() else "missing"
            model, metrics, baseline_metrics, model_source, training_count, testing_count = get_verified_model(dataset_key, model_key)
            score = predict_final_grade(model, *inputs)
    except FileNotFoundError:
        st.error("The project dataset is missing. Restore data/student-por.csv before predicting.")
        return
    except Exception:
        show_error("The prediction service is unavailable. Your inputs have not produced a prediction.")
        return

    # Render the real result into the existing right-hand card.
    with result_content.container():
        st.markdown(f'<div class="result-score">{score:.1f}</div>', unsafe_allow_html=True)
        st.write("An estimated final course grade /20, not a guaranteed academic result.")
        st.caption(model_source)
        if score < 0 or score > 20:
            st.warning("This estimate is outside 0-20. It has not been capped.")
    with st.expander("Model performance & baseline comparison", expanded=False):
        st.caption(f"{training_count:,} training records · {testing_count:,} test records · seed {RANDOM_STATE}")
        for column, name, label in zip(st.columns(3), ('MAE','RMSE','R2'), ('MAE','RMSE','R²')):
            column.metric(label, format_metric(metrics[name]))
        st.table({"Metric": ["MAE", "RMSE", "R²"],
              "Linear Regression": [format_metric(metrics[name]) for name in ('MAE','RMSE','R2')],
              "Training-mean baseline": [format_metric(baseline_metrics[name]) for name in ('MAE','RMSE','R2')]})
        st.write("MAE is average error in score points. RMSE gives more weight to large errors; "
             "lower is better for both. R² measures relative performance, not percentage accuracy; "
             "higher is better, and negative values are possible.")
        st.caption("Undefined R² is N/A. These are held-out results, not percentage accuracy.")
    show_dataset_limitations(report)


def format_metric(value):
    return "N/A" if value is None else f"{value:.3f}"


@st.cache_data(show_spinner=False)
def get_analysis(dataset_fingerprint):
    from gradepredict.analysis import analyze_dataset
    _, _, cleaned = get_project_data(dataset_fingerprint)
    return analyze_dataset(cleaned.dataset)


def show_analysis():
    st.caption("ACADEMIC INSIGHTS / EXPLORATION")
    st.title("Data Analysis")
    st.write("Explore the measurements behind GradePredict AI. Relationships describe associations, not causes.")
    try:
        dataset_key = file_fingerprint(DEFAULT_INPUT_PATH)
        original, report, cleaned = get_project_data(dataset_key)
    except FileNotFoundError:
        st.error("The project dataset is missing. Restore data/student-por.csv to view analysis.")
        return
    except Exception:
        show_error("The dataset could not be loaded and validated.")
        return
    with st.container(key="analysis-facts", border=True):
        columns = st.columns(3)
        columns[0].metric("Student records", f"{len(original.rows):,}")
        columns[1].metric("Dataset columns", len(original.columns))
        columns[2].metric("Records after cleaning", f"{len(cleaned.dataset.rows):,}")
    with st.expander("Dataset overview and quality checks"):
        st.dataframe(list(original.rows[:10]), hide_index=True, width="stretch")
        st.write(f"Exact duplicate records: {len(report.duplicate_records)}. "
                 f"Invalid values flagged: {len(report.invalid_values)}. "
                 f"Missing cells: {sum(report.missing_values.values())}.")
        st.write("Missing values in unused columns are preserved. Missing model inputs are "
                 "imputed from training data only during modeling.")
    try:
        with st.spinner("Preparing statistics and charts…"):
            analysis = get_analysis(dataset_key)
        with st.expander("Descriptive statistics"):
            st.dataframe(analysis['statistics'].round(2), width="stretch")
            st.caption("Cleaned records; missing measurements are excluded without filling them.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gradepredict.visualization import create_figures
        figures = create_figures(cleaned.dataset)
        descriptions = (
            "Compare weekly study-time categories with final course grades; all zero grades remain.",
            "Compare first-period G1 grades with final G3 grades; association is not causation.",
            "Compare second-period G2 grades with final G3 grades.",
            "See the 0-20 final-grade distribution, including zero grades.",
        )
        try:
            st.subheader("Explore the relationships")
            tabs = st.tabs(["Weekly Study Time", "First-period Grade", "Second-period Grade", "Final Grade Distribution"])
            for tab, figure, description in zip(tabs, figures.values(), descriptions):
                with tab:
                    st.caption(description)
                    st.pyplot(figure, width="stretch")
                    if figure is figures['studytime_vs_final_grade']:
                        st.caption("The line inside each box is the median grade; the box covers the middle 50%. "
                                   "Whiskers extend to values within 1.5 box lengths, and separate dots show values beyond them. "
                                   "All records, including zero grades, are included. Differences do not show that study time causes higher grades.")
        finally:
            for figure in figures.values():
                plt.close(figure)
    except Exception:
        show_error("Statistics and charts are unavailable because data-analysis components could not run.")
    show_dataset_limitations(report)


def main():
    st.set_page_config(page_title="GradePredict AI", page_icon="🎓", layout="wide")
    apply_dashboard_style()
    logging.basicConfig(level=logging.INFO)
    with st.sidebar:
        st.subheader("GradePredict AI")
        st.caption("ACADEMIC PERFORMANCE DASHBOARD")
        st.divider()
        page = st.radio("Navigation", PAGES, key="page")
        st.divider()
        st.caption("ARTI303 university project")
    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()
    else:
        show_analysis()


if __name__ == '__main__':
    main()
