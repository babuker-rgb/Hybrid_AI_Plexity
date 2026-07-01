import streamlit as st
import pandas as pd
from pathlib import Path
import zipfile

from pinn_model import (
    load_dataset, PINNTrainer,
    FEATURE_NAMES, OUTPUT_NAMES
)
from nsga2_optimizer import run_nsga2
from plots import plot_pareto, plot_loss_curves, plot_predicted_vs_actual
from report_generator import generate_pdf_report

st.set_page_config(page_title="PINN + NSGA-II Tablet Optimization", layout="wide")
st.title("Hybrid AI Framework: PINN + NSGA-II for Tablet Optimization")

if "df" not in st.session_state:
    st.session_state.df = None
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = None
if "opt_df" not in st.session_state:
    st.session_state.opt_df = None
if "best_formulation" not in st.session_state:
    st.session_state.best_formulation = None

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_synthetic = st.checkbox("Use synthetic data if no CSV", value=True)
    n_samples = st.number_input("Synthetic samples", 1000, 20000, 6000, 500)
    epochs = st.number_input("Training epochs", 50, 2000, 500, 50)
    patience = st.number_input("Patience", 10, 300, 60, 10)
    pop_size = st.number_input("NSGA-II population", 20, 300, 100, 10)
    n_gen = st.number_input("NSGA-II generations", 10, 300, 80, 10)
    run_all = st.button("Run Full Pipeline")

tabs = st.tabs(["Upload", "Train", "Optimize", "Results", "Report", "Download"])

with tabs[0]:
    st.subheader("Dataset")
    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file=uploaded_file)
            st.session_state.df = df
            st.success("CSV loaded successfully.")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(str(e))
    elif use_synthetic:
        st.info("No CSV uploaded. Synthetic dataset will be used.")
    else:
        st.warning("Upload a CSV to proceed.")

with tabs[1]:
    st.subheader("Training")
    if run_all or st.button("Train PINN"):
        try:
            if st.session_state.df is None:
                st.session_state.df = load_dataset(uploaded_file=None, n_samples=n_samples)

            with st.spinner("Training PINN..."):
                trainer = PINNTrainer()
                trainer.fit(st.session_state.df, epochs=epochs, patience=patience)
                st.session_state.trainer = trainer
                st.session_state.metrics_df = trainer.evaluate(st.session_state.df)

                out_dir = Path("output")
                out_dir.mkdir(exist_ok=True)

                st.session_state.metrics_df.to_csv(out_dir / "metrics.csv", index=False)
                plot_loss_curves(trainer.loss_history, out_dir / "loss_curves.png")

            st.success("Training completed.")
            st.dataframe(st.session_state.metrics_df, use_container_width=True)
            st.image(str(out_dir / "loss_curves.png"))
        except Exception as e:
            st.error(str(e))

with tabs[2]:
    st.subheader("Optimization")
    if run_all or st.button("Run NSGA-II"):
        if st.session_state.trainer is None:
            st.warning("Train the PINN first.")
        else:
            try:
                with st.spinner("Running NSGA-II..."):
                    opt_df, best_formulation, res = run_nsga2(
                        st.session_state.trainer,
                        pop_size=pop_size,
                        n_gen=n_gen
                    )
                    st.session_state.opt_df = opt_df
                    st.session_state.best_formulation = best_formulation

                    out_dir = Path("output")
                    out_dir.mkdir(exist_ok=True)

                    opt_df.to_csv(out_dir / "nsga2_pareto_solutions.csv", index=False)
                    plot_pareto(opt_df, out_dir / "pareto_front.png")

                st.success("Optimization completed.")
                st.dataframe(opt_df.head(20), use_container_width=True)
                st.image(str(out_dir / "pareto_front.png"))
            except Exception as e:
                st.error(str(e))

with tabs[3]:
    st.subheader("Results")
    if st.session_state.metrics_df is not None:
        st.markdown("### Model Performance")
        st.dataframe(st.session_state.metrics_df, use_container_width=True)

    if st.session_state.best_formulation is not None:
        st.markdown("### Best Formulation")
        st.dataframe(pd.DataFrame([st.session_state.best_formulation]), use_container_width=True)

    if st.session_state.opt_df is not None:
        st.markdown("### Pareto Solutions")
        st.dataframe(st.session_state.opt_df.head(30), use_container_width=True)

with tabs[4]:
    st.subheader("Report")
    if st.session_state.trainer is not None and st.session_state.opt_df is not None:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)

        y_true = st.session_state.df[OUTPUT_NAMES].values.astype(float)
        y_pred = st.session_state.trainer.predict(st.session_state.df[FEATURE_NAMES].values.astype(float))
        plot_predicted_vs_actual(y_true, y_pred, OUTPUT_NAMES, out_dir / "prediction_plot.png")

        generate_pdf_report(
            out_path=str(out_dir / "report.pdf"),
            best_formulation=st.session_state.best_formulation,
            metrics_df=st.session_state.metrics_df,
            opt_df=st.session_state.opt_df
        )
        st.success("PDF report generated.")
        st.image(str(out_dir / "prediction_plot.png"))
    else:
        st.info("Run training and optimization first.")

with tabs[5]:
    st.subheader("Download")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    if st.session_state.metrics_df is not None and st.session_state.opt_df is not None:
        if st.button("Build ZIP Package"):
            generate_pdf_report(
                out_path=str(out_dir / "report.pdf"),
                best_formulation=st.session_state.best_formulation,
                metrics_df=st.session_state.metrics_df,
                opt_df=st.session_state.opt_df
            )

            zip_path = out_dir / "results_package.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in [
                    "metrics.csv",
                    "nsga2_pareto_solutions.csv",
                    "loss_curves.png",
                    "pareto_front.png",
                    "prediction_plot.png",
                    "report.pdf"
                ]:
                    fpath = out_dir / fname
                    if fpath.exists():
                        zf.write(fpath, arcname=fname)

            st.success("ZIP package created.")

            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download ZIP",
                    data=f.read(),
                    file_name="results_package.zip",
                    mime="application/zip"
                )
    else:
        st.info("No outputs available yet.")
