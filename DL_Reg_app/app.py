import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

# إعداد الصفحة
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# العنوان الرئيسي
st.title("🌞 Solar Power Generation Prediction System")
st.markdown("---")

# تحميل النموذج والمقياس المحفوظين
@st.cache_resource
def load_model_and_scaler():
    import os
    
    # البحث عن الملفات في المسارات المختلفة
    possible_paths = [
        ("DL_Reg_app/my_model.keras", "DL_Reg_app/scaler.pkl"),
        ("my_model.keras", "scaler.pkl"),
        ("./my_model.keras", "./scaler.pkl"),
        ("../my_model.keras", "../scaler.pkl"),
        ("/mount/src/dl_projects/DL_Reg_app/my_model.keras", "/mount/src/dl_projects/DL_Reg_app/scaler.pkl"),
    ]
    
    for model_path, scaler_path in possible_paths:
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = keras.models.load_model(model_path)
                scaler = joblib.load(scaler_path)
                return model, scaler
        except Exception as e:
            continue
    
    st.error("❌ لم يتم العثور على ملفات النموذج. تأكد من وجود my_model.keras و scaler.pkl في المسار الصحيح.")
    return None, None

model, scaler = load_model_and_scaler()

# الشريط الجانبي للتنقل
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("اختر الصفحة", 
                           ["🏠 الرئيسية", "📈 تحليل البيانات", "🔮 التنبؤ", "📊 تقييم النموذج", "🎯 أهمية المتغيرات"])

if page == "🏠 الرئيسية":
    st.header("مرحباً بك في نظام التنبؤ بإنتاج الطاقة الشمسية")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 حول النظام")
        st.write("""
        هذا النظام يستخدم التعلم العميق للتنبؤ بإنتاج الطاقة الشمسية بناءً على:
        - 🌡️ درجة الحرارة
        - 💧 الرطوبة النسبية  
        - ☀️ زاوية السقوط
        - 🌅 زاوية الارتفاع
        - 🌤️ الإشعاع الشمسي
        """)
    
    with col2:
        st.subheader("🎯 الميزات")
        st.write("""
        - ✅ تحليل شامل للبيانات
        - ✅ تنبؤ دقيق بالإنتاج
        - ✅ تقييم أداء النموذج
        - ✅ تحليل أهمية المتغيرات
        - ✅ رسوم بيانية تفاعلية
        """)
    
    # إحصائيات سريعة
    if model is not None:
        st.subheader("📈 إحصائيات النموذج")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("عدد الطبقات", "3", "طبقات كثيفة")
        with col2:
            st.metric("نوع التحسين", "Adam", "محسن متقدم")
        with col3:
            st.metric("دالة الخسارة", "MSE", "متوسط مربع الخطأ")
        with col4:
            st.metric("حالة النموذج", "✅ جاهز", "للاستخدام")

elif page == "📈 تحليل البيانات":
    st.header("📊 تحليل البيانات")
    
    # تحميل البيانات
    uploaded_file = st.file_uploader("ارفع ملف البيانات (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 نظرة عامة على البيانات")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**أول 5 صفوف:**")
                st.dataframe(df.head())
            
            with col2:
                st.write("**معلومات البيانات:**")
                buffer = []
                buffer.append(f"عدد الصفوف: {df.shape[0]}")
                buffer.append(f"عدد الأعمدة: {df.shape[1]}")
                buffer.append(f"القيم المفقودة: {df.isnull().sum().sum()}")
                buffer.append(f"القيم المكررة: {df.duplicated().sum()}")
                st.text("\n".join(buffer))
            
            st.subheader("📈 التوزيع الإحصائي")
            if 'generated_power_kw' in df.columns:
                fig = px.histogram(df, x='generated_power_kw', 
                                 title='توزيع إنتاج الطاقة الشمسية',
                                 labels={'generated_power_kw': 'إنتاج الطاقة (كيلوواط)'})
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔥 خريطة الارتباط")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title="مصفوفة الارتباط بين المتغيرات")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 إحصائيات وصفية")
            st.dataframe(df.describe())
            
        except Exception as e:
            st.error(f"خطأ في قراءة الملف: {e}")
    
    else:
        st.info("📁 يرجى رفع ملف CSV للبدء في تحليل البيانات")

elif page == "🔮 التنبؤ":
    st.header("🔮 تنبؤ إنتاج الطاقة الشمسية")
    
    if model is not None and scaler is not None:
        st.subheader("📝 أدخل البيانات للتنبؤ")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("🌡️ درجة الحرارة (°C)", -20.0, 50.0, 25.0, 0.1)
            humidity = st.slider("💧 الرطوبة النسبية (%)", 0.0, 100.0, 60.0, 0.1)
            solar_zenith = st.slider("🌅 زاوية الارتفاع (درجة)", 0.0, 90.0, 45.0, 0.1)
        
        with col2:
            angle_incidence = st.slider("☀️ زاوية السقوط (درجة)", 0.0, 180.0, 30.0, 0.1)
            solar_radiation = st.slider("☀️ الإشعاع الشمسي (W/m²)", 0.0, 1200.0, 600.0, 1.0)
            wind_speed = st.slider("💨 سرعة الرياح (m/s)", 0.0, 20.0, 5.0, 0.1)
        
        # حساب المتغيرات الإضافية
        delta_angle = abs(angle_incidence - solar_zenith)
        temp_humidity_index = temperature * humidity
        
        st.subheader("📊 المتغيرات المحسوبة")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("فرق الزاوية", f"{delta_angle:.2f}°")
        with col2:
            st.metric("مؤشر الحرارة-الرطوبة", f"{temp_humidity_index:.2f}")
        
        # إعداد البيانات للتنبؤ
        input_data = np.array([[temperature, humidity, solar_zenith, angle_incidence, 
                               solar_radiation, wind_speed, delta_angle, temp_humidity_index]])
        
        if st.button("🎯 تنبؤ الإنتاج", type="primary"):
            try:
                # تطبيق التقييس
                input_scaled = scaler.transform(input_data)
                
                # التنبؤ
                prediction = model.predict(input_scaled)[0][0]
                
                st.success("✅ تم التنبؤ بنجاح!")
                
                # عرض النتيجة
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("⚡ الإنتاج المتوقع", f"{prediction:.2f} kW")
                
                with col2:
                    efficiency = (prediction / 1000) * 100 if prediction > 0 else 0
                    st.metric("📊 الكفاءة المقدرة", f"{efficiency:.1f}%")
                
                with col3:
                    daily_production = prediction * 8  # افتراض 8 ساعات إنتاج
                    st.metric("📅 الإنتاج اليومي", f"{daily_production:.1f} kWh")
                
                # رسم بياني للنتيجة
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "إنتاج الطاقة المتوقع (kW)"},
                    gauge = {
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 250], 'color': "lightgray"},
                            {'range': [250, 500], 'color': "yellow"},
                            {'range': [500, 750], 'color': "orange"},
                            {'range': [750, 1000], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 800
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"خطأ في التنبؤ: {e}")
    
    else:
        st.error("❌ النموذج غير متاح. تأكد من وجود ملفات النموذج.")

elif page == "📊 تقييم النموذج":
    st.header("📊 تقييم أداء النموذج")
    
    # تحميل بيانات الاختبار
    uploaded_file = st.file_uploader("ارفع ملف بيانات الاختبار (CSV)", type=['csv'], key="test_data")
    
    if uploaded_file is not None and model is not None and scaler is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            
            if 'generated_power_kw' in df_test.columns:
                # إعداد البيانات
                y_test = df_test['generated_power_kw']
                X_test = df_test.drop('generated_power_kw', axis=1)
                
                # تطبيق التقييس
                X_test_scaled = scaler.transform(X_test)
                
                # التنبؤ
                y_pred = model.predict(X_test_scaled).flatten()
                
                # حساب المقاييس
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.subheader("📈 مقاييس الأداء")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MSE", f"{mse:.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col3:
                    st.metric("MAE", f"{mae:.2f}")
                with col4:
                    st.metric("R²", f"{r2:.3f}")
                
                # رسم القيم الحقيقية مقابل المتنبأ بها
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = px.scatter(x=y_test, y=y_pred, 
                                     labels={'x': 'القيم الحقيقية', 'y': 'القيم المتنبأ بها'},
                                     title='القيم الحقيقية مقابل المتنبأ بها')
                    fig1.add_trace(px.line(x=[y_test.min(), y_test.max()], 
                                          y=[y_test.min(), y_test.max()]).data[0])
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    residuals = y_test - y_pred
                    fig2 = px.scatter(x=y_test, y=residuals,
                                     labels={'x': 'القيم الحقيقية', 'y': 'البواقي'},
                                     title='رسم البواقي')
                    fig2.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig2, use_container_width=True)
                
                # توزيع البواقي
                st.subheader("📊 تحليل البواقي")
                fig3 = px.histogram(residuals, nbins=30, 
                                   title='توزيع البواقي',
                                   labels={'value': 'البواقي', 'count': 'التكرار'})
                st.plotly_chart(fig3, use_container_width=True)
                
            else:
                st.error("❌ لم يتم العثور على عمود 'generated_power_kw' في البيانات.")
                
        except Exception as e:
            st.error(f"خطأ في تقييم النموذج: {e}")
    
    else:
        if model is None:
            st.error("❌ النموذج غير متاح.")
        else:
            st.info("📁 يرجى رفع ملف بيانات الاختبار لتقييم النموذج.")

elif page == "🎯 أهمية المتغيرات":
    st.header("🎯 تحليل أهمية المتغيرات")
    
    uploaded_file = st.file_uploader("ارفع ملف البيانات لتحليل الأهمية (CSV)", type=['csv'], key="importance_data")
    
    if uploaded_file is not None and model is not None and scaler is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'generated_power_kw' in df.columns:
                st.info("🔄 جاري حساب أهمية المتغيرات... قد يستغرق هذا بعض الوقت.")
                
                # إعداد البيانات
                y = df['generated_power_kw']
                X = df.drop('generated_power_kw', axis=1)
                feature_names = X.columns.tolist()
                
                # تطبيق التقييس
                X_scaled = scaler.transform(X)
                
                # دالة التسجيل
                def r2_scorer(estimator, X, y):
                    y_pred = estimator.predict(X)
                    return r2_score(y, y_pred)
                
                # حساب أهمية التبديل
                perm = permutation_importance(
                    model, X_scaled, y,
                    scoring=r2_scorer,
                    n_repeats=5,  # تقليل العدد لتسريع العملية
                    random_state=42
                )
                
                importance_scores = perm.importances_mean
                importance_normalized = importance_scores / importance_scores.sum()
                
                # إنشاء DataFrame للنتائج
                importance_df = pd.DataFrame({
                    'المتغير': feature_names,
                    'الأهمية': importance_normalized
                }).sort_values('الأهمية', ascending=True)
                
                st.success("✅ تم حساب أهمية المتغيرات بنجاح!")
                
                # رسم بياني للأهمية
                fig = px.bar(importance_df, x='الأهمية', y='المتغير', 
                            orientation='h',
                            title='أهمية المتغيرات في التنبؤ',
                            color='الأهمية',
                            color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # جدول الأهمية
                st.subheader("📋 جدول أهمية المتغيرات")
                importance_display = importance_df.copy()
                importance_display['الأهمية'] = importance_display['الأهمية'].apply(lambda x: f"{x:.3f}")
                st.dataframe(importance_display.sort_values('الأهمية', ascending=False), 
                           use_container_width=True, hide_index=True)
                
                # ملخص الأهمية
                st.subheader("📊 ملخص الأهمية")
                col1, col2, col3 = st.columns(3)
                
                top_features = importance_df.tail(3)['المتغير'].tolist()
                
                with col1:
                    st.metric("🥇 الأهم", top_features[-1])
                with col2:
                    st.metric("🥈 الثاني", top_features[-2])  
                with col3:
                    st.metric("🥉 الثالث", top_features[-3])
                
            else:
                st.error("❌ لم يتم العثور على عمود 'generated_power_kw' في البيانات.")
                
        except Exception as e:
            st.error(f"خطأ في حساب أهمية المتغيرات: {e}")
    
    else:
        if model is None:
            st.error("❌ النموذج غير متاح.")
        else:
            st.info("📁 يرجى رفع ملف البيانات لتحليل أهمية المتغيرات.")

# الهامش السفلي
st.markdown("---")
st.markdown("**تطبيق التنبؤ بإنتاج الطاقة الشمسية** | تم تطويره بواسطة Deep Learning & Streamlit 🚀")
