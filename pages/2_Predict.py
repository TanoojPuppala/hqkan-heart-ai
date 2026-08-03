"""
Streamlit Page 2 — Heart Disease Patient Prediction Engine
Provides complete clinical form inputs, preset patient profiles, instant validation,
Monte Carlo Dropout Bayesian uncertainty estimation, SHAP feature attributions, and PDF report downloads.
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import config
from utils.predictor import load_hqkan_model
from utils.preprocessing import preprocess_patient_input
from utils.uncertainty import predict_with_uncertainty
from utils.shap_utils import calculate_patient_shap, generate_shap_bar_fig
from utils.charts import create_risk_gauge, create_mc_dropout_dist_chart
from utils.report_generator import generate_pdf_report, generate_csv_export


def init_session_defaults():
    """Initializes session state keys for patient input form."""
    if "patient_input" not in st.session_state:
        st.session_state["patient_input"] = {
            "Age": 54,
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 130,
            "Cholesterol": 223,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 140,
            "ExerciseAngina": "N",
            "Oldpeak": 1.0,
            "ST_Slope": "Flat"
        }
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []


def render_predict():
    init_session_defaults()

    st.title("🩺 Heart Disease Risk Prediction Engine")
    st.caption("Input patient diagnostic measurements to perform HQ-KAN Quantum Inference with Bayesian Uncertainty & SHAP Explainability.")

    # Requirement 2: Preset Patient Profile Buttons (High-Risk, Low-Risk, Clear)
    st.subheader("📋 Preset Patient Profiles & Quick Action Tools")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("🔴 Load High-Risk Patient", use_container_width=True):
            st.session_state["patient_input"] = {
                "Age": 63,
                "Sex": "M",
                "ChestPainType": "ASY",
                "RestingBP": 145,
                "Cholesterol": 233,
                "FastingBS": 1,
                "RestingECG": "LVH",
                "MaxHR": 150,
                "ExerciseAngina": "N",
                "Oldpeak": 2.3,
                "ST_Slope": "Down"
            }
            st.toast("Loaded High-Risk Patient profile!", icon="🔴")
            st.rerun()

    with c2:
        if st.button("🟢 Load Low-Risk Patient", use_container_width=True):
            st.session_state["patient_input"] = {
                "Age": 35,
                "Sex": "F",
                "ChestPainType": "ATA",
                "RestingBP": 110,
                "Cholesterol": 180,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 190,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up"
            }
            st.toast("Loaded Low-Risk Patient profile!", icon="🟢")
            st.rerun()

    with c3:
        if st.button("🔄 Clear Form", use_container_width=True):
            st.session_state["patient_input"] = {
                "Age": 54,
                "Sex": "M",
                "ChestPainType": "ASY",
                "RestingBP": 130,
                "Cholesterol": 223,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 140,
                "ExerciseAngina": "N",
                "Oldpeak": 1.0,
                "ST_Slope": "Flat"
            }
            st.toast("Form reset to default neutral parameters.", icon="🔄")
            st.rerun()

    st.divider()

    # Patient Input Form
    st.subheader("📄 Patient Diagnostic Input Form")

    inputs = st.session_state["patient_input"]
    new_input = {}

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("##### 👤 Demographic & History")
        new_input["Age"] = st.slider(
            config.FEATURE_DEFS["Age"]["label"],
            min_value=config.FEATURE_DEFS["Age"]["min"],
            max_value=config.FEATURE_DEFS["Age"]["max"],
            value=int(inputs.get("Age", 54)),
            help=config.FEATURE_DEFS["Age"]["help"]
        )

        sex_opts = config.FEATURE_DEFS["Sex"]["options"]
        sex_idx = sex_opts.index(inputs.get("Sex", "M")) if inputs.get("Sex", "M") in sex_opts else 0
        new_input["Sex"] = st.selectbox(
            config.FEATURE_DEFS["Sex"]["label"],
            options=sex_opts,
            format_func=lambda x: config.FEATURE_DEFS["Sex"]["labels"][x],
            index=sex_idx,
            help=config.FEATURE_DEFS["Sex"]["help"]
        )

        fbs_opts = config.FEATURE_DEFS["FastingBS"]["options"]
        fbs_idx = fbs_opts.index(inputs.get("FastingBS", 0)) if inputs.get("FastingBS", 0) in fbs_opts else 0
        new_input["FastingBS"] = st.selectbox(
            config.FEATURE_DEFS["FastingBS"]["label"],
            options=fbs_opts,
            format_func=lambda x: config.FEATURE_DEFS["FastingBS"]["labels"][x],
            index=fbs_idx,
            help=config.FEATURE_DEFS["FastingBS"]["help"]
        )

    with col_b:
        st.markdown("##### 🫀 Vitals & Electrocardiogram")
        new_input["RestingBP"] = st.slider(
            config.FEATURE_DEFS["RestingBP"]["label"],
            min_value=config.FEATURE_DEFS["RestingBP"]["min"],
            max_value=config.FEATURE_DEFS["RestingBP"]["max"],
            value=int(inputs.get("RestingBP", 130)),
            help=config.FEATURE_DEFS["RestingBP"]["help"]
        )

        new_input["Cholesterol"] = st.slider(
            config.FEATURE_DEFS["Cholesterol"]["label"],
            min_value=config.FEATURE_DEFS["Cholesterol"]["min"],
            max_value=config.FEATURE_DEFS["Cholesterol"]["max"],
            value=int(inputs.get("Cholesterol", 223)),
            help=config.FEATURE_DEFS["Cholesterol"]["help"]
        )

        ecg_opts = config.FEATURE_DEFS["RestingECG"]["options"]
        ecg_idx = ecg_opts.index(inputs.get("RestingECG", "Normal")) if inputs.get("RestingECG", "Normal") in ecg_opts else 0
        new_input["RestingECG"] = st.selectbox(
            config.FEATURE_DEFS["RestingECG"]["label"],
            options=ecg_opts,
            format_func=lambda x: config.FEATURE_DEFS["RestingECG"]["labels"][x],
            index=ecg_idx,
            help=config.FEATURE_DEFS["RestingECG"]["help"]
        )

    with col_c:
        st.markdown("##### 🏃 Exercise Stress & Symptom Test")
        cp_opts = config.FEATURE_DEFS["ChestPainType"]["options"]
        cp_idx = cp_opts.index(inputs.get("ChestPainType", "ASY")) if inputs.get("ChestPainType", "ASY") in cp_opts else 0
        new_input["ChestPainType"] = st.selectbox(
            config.FEATURE_DEFS["ChestPainType"]["label"],
            options=cp_opts,
            format_func=lambda x: config.FEATURE_DEFS["ChestPainType"]["labels"][x],
            index=cp_idx,
            help=config.FEATURE_DEFS["ChestPainType"]["help"]
        )

        new_input["MaxHR"] = st.slider(
            config.FEATURE_DEFS["MaxHR"]["label"],
            min_value=config.FEATURE_DEFS["MaxHR"]["min"],
            max_value=config.FEATURE_DEFS["MaxHR"]["max"],
            value=int(inputs.get("MaxHR", 140)),
            help=config.FEATURE_DEFS["MaxHR"]["help"]
        )

        ang_opts = config.FEATURE_DEFS["ExerciseAngina"]["options"]
        ang_idx = ang_opts.index(inputs.get("ExerciseAngina", "N")) if inputs.get("ExerciseAngina", "N") in ang_opts else 0
        new_input["ExerciseAngina"] = st.selectbox(
            config.FEATURE_DEFS["ExerciseAngina"]["label"],
            options=ang_opts,
            format_func=lambda x: config.FEATURE_DEFS["ExerciseAngina"]["labels"][x],
            index=ang_idx,
            help=config.FEATURE_DEFS["ExerciseAngina"]["help"]
        )

        new_input["Oldpeak"] = st.slider(
            config.FEATURE_DEFS["Oldpeak"]["label"],
            min_value=config.FEATURE_DEFS["Oldpeak"]["min"],
            max_value=config.FEATURE_DEFS["Oldpeak"]["max"],
            step=config.FEATURE_DEFS["Oldpeak"]["step"],
            value=float(inputs.get("Oldpeak", 1.0)),
            help=config.FEATURE_DEFS["Oldpeak"]["help"]
        )

        slope_opts = config.FEATURE_DEFS["ST_Slope"]["options"]
        slope_idx = slope_opts.index(inputs.get("ST_Slope", "Flat")) if inputs.get("ST_Slope", "Flat") in slope_opts else 0
        new_input["ST_Slope"] = st.selectbox(
            config.FEATURE_DEFS["ST_Slope"]["label"],
            options=slope_opts,
            format_func=lambda x: config.FEATURE_DEFS["ST_Slope"]["labels"][x],
            index=slope_idx,
            help=config.FEATURE_DEFS["ST_Slope"]["help"]
        )

    # Update session state with current form values
    st.session_state["patient_input"] = new_input

    st.write("")

    # Predict Button
    predict_clicked = st.button("⚡ Run HQ-KAN Quantum Inference", type="primary", use_container_width=True)

    if predict_clicked or "last_prediction" in st.session_state:

        if predict_clicked:
            # Requirement 3: Input Validation Checks
            if new_input["RestingBP"] < 80:
                st.warning("Resting BP seems too low — should be between 80 and 200 mmHg")
                st.stop()

            if new_input["Cholesterol"] < 100:
                st.warning("Cholesterol seems too low — should be between 100 and 600 mg/dl")
                st.stop()

            if new_input["MaxHR"] < 60:
                st.warning("Max Heart Rate seems too low — should be between 60 and 220 bpm")
                st.stop()

            with st.spinner("Processing Quantum Feature Maps & Running 50 Monte Carlo Dropout Passes..."):
                try:
                    # 1. Preprocess patient inputs
                    x_q_t, x_s_t, df_encoded = preprocess_patient_input(new_input)

                    # 2. Load model
                    model = load_hqkan_model()

                    # 3. Perform Bayesian Uncertainty Inference
                    pred_res = predict_with_uncertainty(model, x_q_t, x_s_t)

                    # 4. Calculate SHAP Explainability
                    shap_res = calculate_patient_shap(x_q_t, x_s_t)

                    st.session_state["last_prediction"] = {
                        "patient_input": new_input.copy(),
                        "pred_res": pred_res,
                        "shap_res": shap_res,
                        "x_q_t": x_q_t,
                        "x_s_t": x_s_t
                    }

                    # Add to session history log
                    hist_item = {
                        "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                        "Age": new_input["Age"],
                        "Sex": new_input["Sex"],
                        "Probability": f"{pred_res['mean_prob']*100:.1f}%",
                        "Risk Level": pred_res["risk_level"],
                        "Confidence": f"{pred_res['confidence_pct']:.1f}%"
                    }
                    st.session_state["prediction_history"].append(hist_item)

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    return

        # Render Prediction Results
        res = st.session_state["last_prediction"]
        pred_res = res["pred_res"]
        shap_res = res["shap_res"]
        patient = res["patient_input"]

        st.divider()
        st.subheader("📊 Quantum Diagnostic Output & Bayesian Risk Assessment")

        res_col1, res_col2 = st.columns([1.2, 1.0])

        with res_col1:
            if pred_res["is_disease"]:
                st.markdown(f"""
                <div class="med-card-disease">
                    <h2 style="color: #9B1C1C; margin-bottom: 4px;">🚨 HIGH RISK: HEART DISEASE DETECTED</h2>
                    <h1 style="color: #E74C3C; font-size: 3rem; margin: 8px 0;">{pred_res['mean_prob']*100:.1f}%</h1>
                    <p style="color: #7F1D1D; font-size: 1rem; font-weight: 600;">
                        Confidence Score: <b>{pred_res['confidence_pct']:.1f}%</b> | Uncertainty Margin: <b>±{pred_res['uncertainty_std']*100:.2f}%</b>
                    </p>
                    <span class="status-pill status-high">{pred_res['risk_level']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="med-card-healthy">
                    <h2 style="color: #03543F; margin-bottom: 4px;">💚 LOW RISK: PATIENT HEALTHY</h2>
                    <h1 style="color: #2ECC71; font-size: 3rem; margin: 8px 0;">{(1-pred_res['mean_prob'])*100:.1f}% Healthy</h1>
                    <p style="color: #064E3B; font-size: 1rem; font-weight: 600;">
                        Confidence Score: <b>{pred_res['confidence_pct']:.1f}%</b> | Uncertainty Margin: <b>±{pred_res['uncertainty_std']*100:.2f}%</b>
                    </p>
                    <span class="status-pill status-low">{pred_res['risk_level']}</span>
                </div>
                """, unsafe_allow_html=True)

            # Uncertainty score clinical interpretation box
            unc_pct = pred_res["uncertainty_std"] * 100
            if unc_pct < 10.0:
                st.info("ℹ️ **Uncertainty Interpretation**: Model is highly confident in this prediction. The quantum circuit produced consistent outputs across all 50 Monte Carlo samples. This result can be used with high clinical confidence.")
            elif unc_pct <= 20.0:
                st.warning("⚠️ **Uncertainty Interpretation**: Moderate uncertainty detected. The model shows some variability across Monte Carlo samples. Consider ordering additional tests such as echocardiogram or stress test before making a final clinical decision.")
            else:
                st.error("🚨 **Uncertainty Interpretation**: High uncertainty detected. The model is not confident about this prediction. This patient may have atypical feature combinations that the model has not seen frequently during training. Additional clinical investigation is strongly recommended.")

        with res_col2:
            st.plotly_chart(create_risk_gauge(pred_res["mean_prob"], pred_res["uncertainty_std"], pred_res["is_disease"]), use_container_width=True)

        st.write("")

        # Requirement 4: Visual Risk Meter Progress Bar
        prob_pct = pred_res["mean_prob"] * 100
        if pred_res["mean_prob"] < 0.30:
            bar_color = "#2ECC71"
            risk_label = "LOW RISK"
        elif pred_res["mean_prob"] <= 0.60:
            bar_color = "#F39C12"
            risk_label = "MODERATE RISK"
        else:
            bar_color = "#E74C3C"
            risk_label = "HIGH RISK"

        bar_text = f"{risk_label} ({prob_pct:.1f}%)"

        st.markdown(f"""
        <div style="width: 100%; background-color: #E2E8F0; border-radius: 10px; height: 30px; position: relative; margin-top: 15px; margin-bottom: 20px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
            <div style="width: {max(prob_pct, 18.0):.1f}%; background-color: {bar_color}; height: 30px; border-radius: 10px; transition: width 0.6s ease; display: flex; align-items: center; justify-content: center;">
                <span style="color: #FFFFFF; font-weight: bold; font-size: 0.9rem; white-space: nowrap; padding: 0 10px; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">{bar_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # SHAP & Uncertainty Distribution Section
        st.subheader("💡 SHAP Feature Attribution & Uncertainty Distribution")
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            fig_shap = generate_shap_bar_fig(shap_res["shap_values"], shap_res["feature_names"])
            st.pyplot(fig_shap)

        with s_col2:
            st.plotly_chart(create_mc_dropout_dist_chart(pred_res["sample_preds"]), use_container_width=True)

            st.markdown("##### 🔍 Top Drivers For This Patient")
            pos_str = ", ".join([f"**{k}** (+{v:.2f})" for k, v, _ in shap_res["top_positive"][:3]])
            neg_str = ", ".join([f"**{k}** ({v:.2f})" for k, v, _ in shap_res["top_negative"][:3]])
            
            st.write(f"• **Risk Elevating Features**: {pos_str if pos_str else 'None'}")
            st.write(f"• **Protective Features**: {neg_str if neg_str else 'None'}")

        st.divider()

        # Download Options
        st.subheader("📥 Export Diagnostic Reports")
        d1, d2 = st.columns(2)

        with d1:
            pdf_bytes = generate_pdf_report(patient, pred_res)
            st.download_button(
                label="📄 Download PDF Clinical Diagnostic Report",
                data=pdf_bytes,
                file_name=f"HQKAN_Patient_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with d2:
            csv_str = generate_csv_export(patient, pred_res)
            st.download_button(
                label="📊 Download CSV Patient Record",
                data=csv_str,
                file_name=f"HQKAN_Patient_Record_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Session History Table
    if st.session_state["prediction_history"]:
        st.divider()
        st.subheader("📜 Session Prediction History")
        df_hist = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(df_hist, use_container_width=True)


if __name__ == "__main__":
    render_predict()
