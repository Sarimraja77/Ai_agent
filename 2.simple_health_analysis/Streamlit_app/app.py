from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Blood Report", page_icon=":guardsman:", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it")

st.markdown("""
<style>
.scroll-box{
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
    background-color: #1e1e1e;
    font-size: 0.9rem;
    line-height: 1.6;
}
.scroll-box p, .scroll-box li {
    color: #e0e0e0;
}
.section-label {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #ffffff;
}
</style>
""",unsafe_allow_html=True)
st.title("Blood Report Analysis")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Blood Report")
    blood_report = st.text_area(
        label = "Enter your blood report details here:",
        height=500,
        placeholder="place your blood report here",
        label_visibility="collapsed"
    )
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

with right_col:
    st.subheader("Health Summary")
    health_box = st.empty()
    health_box.markdown('<div class="scroll-box"></div>', unsafe_allow_html=True)

    st.subheader("Diet Recommendations")
    diet_box = st.empty()
    diet_box.markdown('<div class="scroll-box"></div>', unsafe_allow_html=True)

if analyze_clicked:
    if not blood_report.strip():
        with left_col:
            st.warning("Please enter your blood report details before analyzing.")    
    else:
        with st.spinner("Analyzing your blood report..."):

            extraction_prompt = f"""You are a medical data extraction assistant.

From the blood report below, extract all test values and classify each one as High,Low or normal
based on the refrence ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""
            extraction_response = llm.invoke(extraction_prompt)
            extracted_values = extraction_response.text

            diet_prompt = f"""
Your are a clinical nutritionist specializing in indian dietary habits.

Based on the following blood report analysis, provide two clearly seperated sections:

SECTION 1: Health Summary:
Write 4-5 lines explaining the patient's condition in simple, non-technical language. 

SECTION 2: Diet Recommendations:
List foods to eat more of and foods to avoid, using commonly available Indian foods
like dal, sabzi, roti, rice, etc. Keep it practical and concise.

Blood Report Analysis:
{extracted_values}
"""
            diet_response = llm.invoke(diet_prompt)
            full_response = diet_response.text

        if "SECTION 2" in full_response:
            parts = full_response.split("SECTION 2")
            health_summary = parts[0].replace("SECTION 1 - HEALTH SUMMARY:", "").replace("SECTION 1", "").strip()
            diet_plan = ("SECTION 2" + parts[1]).replace("SECTION 2 - INDIAN DIET PLAN:", "").replace("SECTION 2", "").strip()
        else:
            health_summary = full_response
            diet_plan = ""
        
        # Render into fixed-height scrollable boxes
        health_box.markdown(
            f'<div class="scroll-box">{health_summary}</div>',
            unsafe_allow_html=True
                )
        diet_box.markdown(
            f'<div class="scroll-box">{diet_plan if diet_plan else full_response}</div>',
            unsafe_allow_html=True
        )
        