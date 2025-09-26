import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# إعداد الصفحة
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide"
)

# العنوان الرئيسي
st.title("☀️ Solar Power Generation Predictor")
st.markdown("---")

# دالة لتحميل النموذج والـ scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = keras.models.load_model("my_model.keras")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None, None

# تحميل النموذج والـ scaler
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("لم يتم العثور على ملفات النموذج. تأكد من وجود my_model.keras و scaler.pkl في نفس المجلد")
    st.stop()

# إنشاء أعمدة للإدخال
st.sidebar.header("إعدادات المدخلات")

# معلومات حول البيانات المطلوبة (يجب تعديلها حسب البيانات الفعلية)
st.sidebar.info("قم بإدخال القيم التالية للحصول على توقع الطاقة المولدة")

# إنشاء مدخلات للمتغيرات (يجب تعديل هذه المدخلات حسب الأعمدة الفعلية في بياناتك)
# سأضع أمثلة شائعة للمتغيرات في أنظمة الطاقة الشمسية

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 إدخال البيانات")
    
    # مدخلات افتراضية - يجب تعديلها حسب بياناتك الفعلية
    solar_irradiance = st.number_input(
        "الإشعاع الشمسي (W/m²)", 
        min_value=0.0, 
        max_value=1500.0, 
        value=800.0,
        step=10.0
    )
    
    temperature = st.number_input(
        "درجة الحرارة (°C)", 
        min_value=-20.0, 
        max_value=60.0, 
        value=25.0,
        step=0.1
    )
    
    humidity = st.number_input(
        "الرطوبة (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0,
        step=1.0
    )
    
    wind_speed = st.number_input(
        "سرعة الرياح (m/s)", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0,
        step=0.1
    )
    
    # يمكن إضافة المزيد من المتغيرات حسب بياناتك
    panel_area = st.number_input(
        "مساحة الألواح (m²)", 
        min_value=1.0, 
        max_value=1000.0, 
        value=50.0,
        step=1.0
    )

with col2:
    st.subheader("🔮 النتائج")
    
    if st.button("توقع الطاقة المولدة", type="primary"):
        try:
            # إنشاء مصفوفة الإدخال (يجب تعديلها حسب عدد المتغيرات الفعلي)
            input_data = np.array([[
                solar_irradiance,
                temperature,
                humidity,
                wind_speed,
                panel_area
                # أضف المزيد من المتغيرات حسب بياناتك
            ]])
            
            # تطبيق التطبيع
            input_scaled = scaler.transform(input_data)
            
            # التنبؤ
            prediction = model.predict(input_scaled)
            predicted_power = prediction[0][0]
            
            # عرض النتيجة
            st.success(f"الطاقة المتوقعة: **{predicted_power:.2f} kW**")
            
            # إنشاء مخطط دائري للنتيجة
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = predicted_power,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "الطاقة المولدة (kW)"},
                gauge = {
                    'axis': {'range': [None, max(500, predicted_power * 1.5)]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, predicted_power * 0.5], 'color': "lightgray"},
                        {'range': [predicted_power * 0.5, predicted_power * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': predicted_power * 0.9
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"خطأ في التنبؤ: {e}")

# قسم معلومات إضافية
st.markdown("---")
st.subheader("📈 معلومات إضافية")

col3, col4 = st.columns(2)

with col3:
    st.info("""
    **حول النموذج:**
    - نوع النموذج: شبكة عصبية عميقة
    - عدد الطبقات: 3 طبقات مخفية
    - دالة التفعيل: ReLU
    - المحسن: Adam
    """)

with col4:
    st.warning("""
    **ملاحظات مهمة:**
    - تأكد من دقة البيانات المدخلة
    - النتائج تعتمد على جودة بيانات التدريب
    - قد تختلف النتائج حسب الظروف البيئية
    """)

# قسم رفع ملف CSV للتنبؤ المتعدد
st.markdown("---")
st.subheader("📁 التنبؤ من ملف CSV")

uploaded_file = st.file_uploader(
    "ارفع ملف CSV للتنبؤ المتعدد", 
    type=['csv']
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("عينة من البيانات:")
        st.dataframe(df_input.head())
        
        if st.button("تنفيذ التنبؤ على الملف"):
            # تطبيق التطبيع
            X_scaled = scaler.transform(df_input)
            
            # التنبؤ
            predictions = model.predict(X_scaled)
            
            # إضافة التنبؤات إلى البيانات
            df_results = df_input.copy()
            df_results['predicted_power_kw'] = predictions
            
            st.success("تم التنبؤ بنجاح!")
            st.dataframe(df_results)
            
            # إنشاء مخطط للنتائج
            fig = px.line(
                df_results, 
                y='predicted_power_kw',
                title="التنبؤات عبر الزمن",
                labels={'predicted_power_kw': 'الطاقة المتوقعة (kW)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # خيار تحميل النتائج
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="تحميل النتائج كملف CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"خطأ في معالجة الملف: {e}")

# معلومات التطبيق
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Solar Power Generation Predictor v1.0</p>
    <p>Built with Streamlit and TensorFlow</p>
</div>
""", unsafe_allow_html=True)
