from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import dataprocessing  # Required so joblib can unpickle FunctionTransformer(engineer_features)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
)

if not (MODELS_DIR / "churn_model.joblib").exists():
    st.error("⚠️ مفيش موديل متسيف. شغّل الأول في التيرمينال:  `python train.py`")
    st.stop()


@st.cache_resource
def load_artifacts():
    """Load model and results once and cache in memory across UI reruns."""
    loaded_model = joblib.load(MODELS_DIR / "churn_model.joblib")
    loaded_name = (MODELS_DIR / "best_model_name.txt").read_text().strip()
    loaded_results = pd.read_csv(MODELS_DIR / "results.csv")
    return loaded_model, loaded_name, loaded_results


model, best_name, results_df = load_artifacts()
best_row = results_df.loc[results_df["Model"] == best_name].iloc[0]

# ---------------- Sidebar: Model Summary ----------------
with st.sidebar:
    st.header("🤖 معلومات الموديل الحالي")
    st.success(f"**Best Model:** {best_name}")
    st.caption("تم اختياره تلقائياً بناءً على أعلى F1-Score على بيانات الاختبار.")

    s_col1, s_col2 = st.columns(2)
    s_col1.metric("F1-Score", f"{best_row['F1']:.1%}")
    s_col2.metric("ROC-AUC", f"{best_row['ROC-AUC']:.1%}")

    s_col3, s_col4 = st.columns(2)
    s_col3.metric("Recall", f"{best_row['Recall']:.1%}")
    s_col4.metric("Precision", f"{best_row['Precision']:.1%}")

    st.divider()
    st.caption("📊 **خصائص إضافية بيحسبها الـ Pipeline تلقائياً:**")
    st.markdown(
        "- `TotalServices`: إجمالي عدد الخدمات المشترك فيها\n"
        "- `AvgMonthlyCharges`: متوسط الدفع الشهري الفعلي\n"
        "- `ChargeDifference`: فرق الفاتورة الحالية عن المتوسط"
    )

# ---------------- Main Header ----------------
st.title("📞 Telco Customer Churn Predictor")
st.caption("نظام ذكي لتوقع احتمالية مغادرة العملاء ومقارنة أداء خوارزميات التصنيف")

tab1, tab2 = st.tabs(["🔍 توقع عميل جديد (Predict)", "📊 مقارنة الموديلات (Models Comparison)"])

# ---------------- Tab 1: Prediction Form ----------------
with tab1:
    col_profile, col_services, col_billing = st.columns(3)

    # Card 1: Customer Profile
    with col_profile:
        with st.container(border=True):
            st.subheader("👤 بيانات العميل")
            gender = st.segmented_control("Gender", ["Female", "Male"], default="Female") or "Female"
            senior = st.segmented_control("Senior Citizen", ["No", "Yes"], default="No") or "No"
            partner = st.segmented_control("Partner", ["No", "Yes"], default="No") or "No"
            dependents = st.segmented_control("Dependents", ["No", "Yes"], default="No") or "No"
            tenure = st.slider("Tenure (عدد شهور الاشتراك)", 0, 72, 12)

    # Card 2: Subscribed Services
    with col_services:
        with st.container(border=True):
            st.subheader("🌐 الخدمات المشترك بها")
            phone = st.selectbox("Phone Service", ["Yes", "No"])

            if phone == "No":
                multiple = "No phone service"
                st.info("ℹ️ خدمة الخطوط المتعددة غير متاحة لعدم وجود خدمة هاتف.")
            else:
                multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            if internet == "No":
                security = backup = protection = support = tv = movies = "No internet service"
                st.info("ℹ️ خدمات الإنترنت الإضافية غير مفعلة لأن العميل غير مشترك في الإنترنت.")
            else:
                st.caption("الخدمات الإضافية للإنترنت:")
                c1, c2 = st.columns(2)
                with c1:
                    security = "Yes" if st.checkbox("Online Security") else "No"
                    backup = "Yes" if st.checkbox("Online Backup") else "No"
                    protection = "Yes" if st.checkbox("Device Protection") else "No"
                with c2:
                    support = "Yes" if st.checkbox("Tech Support") else "No"
                    tv = "Yes" if st.checkbox("Streaming TV") else "No"
                    movies = "Yes" if st.checkbox("Streaming Movies") else "No"

    # Card 3: Contract & Billing
    with col_billing:
        with st.container(border=True):
            st.subheader("💳 التعاقد والفواتير")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
            default_total = 0.0 if tenure == 0 else float(monthly * tenure)
            total = st.number_input(
                "Total Charges ($)",
                0.0,
                10000.0,
                default_total,
                step=50.0,
                disabled=(tenure == 0),
            )

    # Predict Button
    if st.button("🚀 احسب احتمالية ترك الخدمة (Predict Churn)", type="primary", use_container_width=True):
        actual_total = 0.0 if tenure == 0 else total
        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": actual_total,
        }])

        prob = float(model.predict_proba(row)[0][1])
        engineered = dataprocessing.engineer_features(row).iloc[0]

        with st.container(border=True):
            res_col1, res_col2 = st.columns([1.2, 1.8])
            with res_col1:
                st.subheader("🎯 نتيجة التوقع")
                st.metric("احتمالية المغادرة (Churn Probability)", f"{prob:.1%}")
                st.progress(prob)

                if prob >= 0.65:
                    st.error("🚨 **خطورة عالية (High Risk):** العميل ده غالباً هيسيب الشركة قريب جداً — يفضل التواصل معاه فوراً وتقديم عرض استبقاء!")
                elif prob >= 0.50:
                    st.warning("⚠️ **خطورة متوسطة (Medium Risk):** العميل معرض لترك الخدمة — يُنصح بمتابعة رضاه أو اقتراح عقد سنوي.")
                else:
                    st.success("✅ **عميل مستقر (Low Risk):** العميل ده غالباً هيكمّل معانا ومستقر في اشتراكه.")

            with res_col2:
                st.subheader("⚙️ الخصائص المستنتجة تلقائياً (Engineered Features)")
                st.caption("القيم دي تم حسابها داخل الـ Pipeline من بيانات العميل قبل دخولها للموديل:")
                f1, f2, f3 = st.columns(3)
                f1.metric(
                    "إجمالي الخدمات (TotalServices)",
                    f"{int(engineered['TotalServices'])} / 7",
                )
                f2.metric(
                    "متوسط الدفع الفعلي (AvgMonthly)",
                    f"${engineered['AvgMonthlyCharges']:.1f}",
                )
                f3.metric(
                    "فرق الفاتورة الحالية (ChargeDiff)",
                    f"${engineered['ChargeDifference']:+.1f}",
                )

# ---------------- Tab 2: Comparison ----------------
with tab2:
    st.subheader("🏆 ترتيب الموديلات حسب الأداء")

    top_f1 = results_df.sort_values("F1", ascending=False).iloc[0]
    top_auc = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]
    top_rec = results_df.sort_values("Recall", ascending=False).iloc[0]

    k1, k2, k3 = st.columns(3)
    with k1:
        with st.container(border=True):
            st.metric("🥇 أعلى F1-Score", top_f1["Model"], f"{top_f1['F1']:.1%}")
    with k2:
        with st.container(border=True):
            st.metric("📈 أعلى ROC-AUC", top_auc["Model"], f"{top_auc['ROC-AUC']:.1%}")
    with k3:
        with st.container(border=True):
            st.metric("🎯 أعلى Recall", top_rec["Model"], f"{top_rec['Recall']:.1%}")

    st.dataframe(
        results_df.round(3),
        use_container_width=True,
        hide_index=True,
    )

    chart_cols = [c for c in ["F1", "CV_F1", "ROC-AUC", "Recall", "Precision"] if c in results_df.columns]
    selected_metrics = st.multiselect(
        "اختر المقاييس لعرضها في الرسم البياني:",
        options=chart_cols,
        default=[c for c in ["F1", "CV_F1", "ROC-AUC"] if c in chart_cols],
    )
    if selected_metrics:
        st.bar_chart(results_df.set_index("Model")[selected_metrics])