import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from fpdf import FPDF
import base64

# Load and display the logo
logo = Image.open("ttu_logo.png")
st.image(logo, width=900)

# Header
st.markdown("""
<h2 style='text-align: center; color: navy;'>Graduation Project II</h2>
<h3 style='text-align: center; color: darkgreen;'>College of Engineering / Natural Resources and Chemical Engineering Department</h3>
<h4 style='text-align: center;'>Tafila Technical University</h4>
<h5 style='text-align: center; color: gray;'>Designed and implemented by students:</h5>
<ul style='text-align: center; list-style: none; padding-left: 0;'>
    <li>1 - Duaa</li>
    <li>2 - Shahed</li>
    <li>3 - Rahaf</li>
</ul>
""", unsafe_allow_html=True)

# Load model and scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
model = load_model('ann_water_model.h5')

# Output variable labels
output_vars = [
    'Turbidity_final_NTU', 'Fe_final_mg_L', 'Mn_final_mg_L', 'Cu_final_mg_L',
    'Zn_final_mg_L', 'Suspended_solids_final_mg_L', 'TDS_final_mg_L',
    'Turbidity_removal_%', 'Suspended_solids_removal_%', 'TDS_removal_%'
]

# Standard reuse limits (with signs)
limits = {
    'Turbidity_final_NTU': ('<', 5.0),
    'Fe_final_mg_L': ('<', 0.3),
    'Mn_final_mg_L': ('<', 0.1),
    'Cu_final_mg_L': ('<', 1.0),
    'Zn_final_mg_L': ('<', 5.0),
    'Suspended_solids_final_mg_L': ('<', 50.0),
    'TDS_final_mg_L': ('<', 1000.0)
}

# UI
st.title("Water Treatment Quality Predictor (ANN-based)")
st.markdown("**Enter experimental values below to predict treated water quality and assess reuse suitability.**")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pH_raw = st.slider("pH of Raw Water", 3.0, 11.0, 7.0)
        turbidity_raw = st.slider("Turbidity (NTU)", 0.1, 500.0, 50.0)
        temperature = st.slider("Temperature (°C)", 5.0, 40.0, 25.0)
        coagulant_dose = st.slider("Coagulant Dose (mg/L)", 0.0, 100.0, 30.0)
        flocculant_dose = st.slider("Flocculant Dose (mg/L)", 0.0, 20.0, 5.0)
        fe_initial = st.slider("Initial Fe (mg/L)", 0.0, 10.0, 1.0)
        mn_initial = st.slider("Initial Mn (mg/L)", 0.0, 5.0, 0.3)

    with col2:
        cu_initial = st.slider("Initial Cu (mg/L)", 0.0, 2.0, 0.05)
        zn_initial = st.slider("Initial Zn (mg/L)", 0.0, 5.0, 0.1)
        ss = st.slider("Suspended Solids (mg/L)", 0.0, 1000.0, 150.0)
        tds = st.slider("TDS (mg/L)", 0.0, 5000.0, 1000.0)
        mixing_speed = st.slider("Mixing Speed (rpm)", 50, 500, 200)
        rapid_mix = st.slider("Rapid Mix Time (min)", 0.5, 10.0, 2.0)
        slow_mix = st.slider("Slow Mix Time (min)", 1.0, 30.0, 10.0)
        settling_time = st.slider("Settling Time (min)", 5.0, 60.0, 30.0)

    submitted = st.form_submit_button("Test Water Quality")

# Prediction
if submitted:
    input_array = np.array([[pH_raw, turbidity_raw, temperature, coagulant_dose, flocculant_dose,
                             fe_initial, mn_initial, cu_initial, zn_initial, ss, tds,
                             mixing_speed, rapid_mix, slow_mix, settling_time]])
    
    X_scaled = scaler_X.transform(input_array)
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

    # Results formatting
    results = pd.DataFrame([y_pred], columns=output_vars).T
    results.columns = ['Predicted Value']
    results['Unit'] = ['NTU', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', 'mg/L', '%', '%', '%']

    assessments = []
    standards = []
    reuse_safe = True

    for i, var in enumerate(output_vars):
        if var in limits:
            sign, limit = limits[var]
            value = y_pred[i]
            standards.append(f"{sign} {limit}")
            if sign == '<' and value > limit:
                assessments.append("❌ Exceeds Limit")
                reuse_safe = False
            else:
                assessments.append("✅ OK")
        else:
            standards.append("--")
            assessments.append("--")

    results['Standard Limit'] = standards
    results['Assessment'] = assessments

    st.subheader("Predicted Treated Water Quality")
    st.dataframe(results.style.format(precision=3))

    st.subheader("Reuse Decision")
    if reuse_safe:
        st.success("✅ Water is safe for reuse or discharge.")
    else:
        st.error("❌ Water does NOT meet quality standards for reuse.")

    # PDF generation
    def create_pdf(results_df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Water Quality Prediction Report", ln=True, align='C')
        pdf.ln(10)
        for index, row in results_df.iterrows():
            line = f"{index}: {row['Predicted Value']:.3f} {row['Unit']} | Limit: {row['Standard Limit']} | {row['Assessment']}"
            pdf.cell(200, 10, txt=line, ln=True)
        return pdf.output(dest='S').encode('latin1')

    if st.button("📄 Download Report as PDF"):
        pdf_data = create_pdf(results)
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="Water_Quality_Report.pdf">📥 Click here to download your report</a>'
        st.markdown(href, unsafe_allow_html=True)
