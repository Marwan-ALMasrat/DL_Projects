import streamlit as st
import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    st.error("TensorFlow is not available. Please install TensorFlow.")
    TF_AVAILABLE = False
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
        # البحث عن الملفات في مسارات مختلفة
        model_paths = [
            "my_model.keras",
            "DL_Reg_app/my_model.keras",
            "./DL_Reg_app/my_model.keras"
        ]
        
        scaler_paths = [
            "scaler.pkl",
            "DL_Reg_app/scaler.pkl", 
            "./DL_Reg_app/scaler.pkl"
        ]
        
        model = None
        scaler = None
        
        # جرب تحميل النموذج من المسارات المختلفة
        for model_path in model_paths:
            try:
                model = keras.models.load_model(model_path)
                break
            except:
                continue
                
        # جرب تحميل الـ scaler من المسارات المختلفة
        for scaler_path in scaler_paths:
            try:
                scaler = joblib.load(scaler_path)
                break
            except:
                continue
                
        return model, scaler
        
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {e}")
        return None, None

# تحميل النموذج والـ scaler
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("لم يتم العثور على ملفات النموذج. تأكد من وجود my_model.keras و scaler.pkl في نفس المجلد")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # خيار إنشاء نموذج تجريبي
        if st.button("🚀 إنشاء نموذج تجريبي", type="primary"):
            with st.spinner("جاري إنشاء نموذج تجريبي..."):
                try:
                    # إنشاء نموذج بسيط للاختبار
                    demo_model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(5,)),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(1)
                    ])
                    demo_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    # إنشاء بيانات وهمية واقعية أكثر
                    np.random.seed(42)
                    n_samples = 1000
                    
                    # متغيرات واقعية للطاقة الشمسية
                    solar_irradiance = np.random.uniform(100, 1200, n_samples)
                    temperature = np.random.uniform(15, 45, n_samples)
                    humidity = np.random.uniform(20, 90, n_samples)
                    wind_speed = np.random.uniform(0, 15, n_samples)
                    panel_area = np.random.uniform(10, 100, n_samples)
                    
                    X_dummy = np.column_stack([solar_irradiance, temperature, humidity, wind_speed, panel_area])
                    
                    # معادلة واقعية للطاقة المولدة
                    efficiency = 0.2  # كفاءة 20%
                    temp_coefficient = -0.004  # معامل درجة الحرارة
                    
                    y_dummy = (
                        solar_irradiance * panel_area * efficiency * 
                        (1 + temp_coefficient * (temperature - 25)) * 
                        (1 - humidity * 0.001) * 
                        (1 + wind_speed * 0.01)
                    ) / 1000  # تحويل إلى kW
                    
                    # إضافة بعض الضوضاء
                    y_dummy += np.random.normal(0, y_dummy.std() * 0.1, n_samples)
                    y_dummy = np.maximum(y_dummy, 0)  # التأكد من عدم وجود قيم سالبة
                    
                    demo_model.fit(X_dummy, y_dummy, epochs=50, verbose=0, validation_split=0.2)
                    demo_model.save("my_model.keras")
                    
                    # إنشاء scaler
                    from sklearn.preprocessing import StandardScaler
                    demo_scaler = StandardScaler()
                    demo_scaler.fit(X_dummy)
                    joblib.dump(demo_scaler, "scaler.pkl")
                    
                    st.success("✅ تم إنشاء النموذج التجريبي بنجاح!")
                    st.info("يرجى إعادة تحميل الصفحة لاستخدام النموذج الجديد")
                    
                    # إعادة تحميل النموذج
                    st.cache_resource.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ خطأ في إنشاء النموذج التجريبي: {e}")
    
    with col2:
        # خيار رفع ملف البيانات لإنشاء نموذج حقيقي
        st.markdown("### 📊 أو ارفع بيانات التدريب")
        uploaded_train_file = st.file_uploader(
            "ارفع ملف CSV لبيانات التدريب", 
            type=['csv'],
            help="ملف يحتوي على البيانات مع عمود 'generated_power_kw'"
        )
        
        if uploaded_train_file is not None:
            if st.button("🎯 تدريب نموذج من البيانات"):
                with st.spinner("جاري تدريب النموذج..."):
                    try:
                        # قراءة البيانات
                        df_train = pd.read_csv(uploaded_train_file)
                        
                        if 'generated_power_kw' not in df_train.columns:
                            st.error("الملف يجب أن يحتوي على عمود 'generated_power_kw'")
                        else:
                            # إعداد البيانات
                            y = df_train['generated_power_kw']
                            X = df_train.drop('generated_power_kw', axis=1)
                            
                            # تقسيم البيانات
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # التطبيع
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # بناء النموذج
                            model_real = tf.keras.Sequential([
                                tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                                tf.keras.layers.Dense(128, activation='relu'),
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(1)
                            ])
                            
                            model_real.compile(optimizer='adam', loss='mse', metrics=['mae'])
                            
                            # التدريب
                            history = model_real.fit(
                                X_train_scaled, y_train,
                                validation_data=(X_test_scaled, y_test),
                                epochs=100,
                                verbose=0,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(
                                        patience=10, restore_best_weights=True
                                    )
                                ]
                            )
                            
                            # حفظ النموذج والـ scaler
                            model_real.save("my_model.keras")
                            joblib.dump(scaler, "scaler.pkl")
                            
                            # عرض النتائج
                            train_loss = history.history['loss'][-1]
                            val_loss = history.history['val_loss'][-1]
                            
                            st.success("✅ تم تدريب النموذج بنجاح!")
                            st.write(f"**خسارة التدريب:** {train_loss:.4f}")
                            st.write(f"**خسارة التحقق:** {val_loss:.4f}")
                            
                            st.cache_resource.clear()
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ خطأ في تدريب النموذج: {e}")
    
    st.markdown("---")
    st.info("""
    💡 **إرشادات:**
    - استخدم النموذج التجريبي للاختبار السريع
    - لنموذج حقيقي، ارفع ملف CSV يحتوي على بياناتك مع عمود 'generated_power_kw'
    - تأكد من أن البيانات تحتوي على المتغيرات: الإشعاع الشمسي، درجة الحرارة، الرطوبة، سرعة الرياح، مساحة الألواح
    """)
    
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
