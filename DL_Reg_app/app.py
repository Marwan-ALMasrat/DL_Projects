import streamlit as st
import pandas as pd
import numpy as np
import joblib
import keras
import plotly.graph_objects as go

# إعداد الصفحة
st.set_page_config(
    page_title="Solar Power Generation Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# العنوان الرئيسي
st.title("🌞 Solar Power Generation Prediction System")
st.markdown("### أدخل البيانات البيئية للحصول على تنبؤ دقيق لإنتاج الطاقة الشمسية")
st.markdown("---")

if model is not None and scaler is not None:
    
    # تنظيم الإدخالات في أعمدة منطقية
    st.subheader("🌡️ البيانات البيئية الأساسية")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌡️ درجة الحرارة والرطوبة**")
        temperature_2_m_above_gnd = st.slider(
            "درجة الحرارة على ارتفاع 2م (°C)", 
            -30.0, 60.0, 25.0, 0.5,
            help="درجة الحرارة المحيطة تؤثر على كفاءة الألواح الشمسية"
        )
        
        relative_humidity_2_m_above_gnd = st.slider(
            "الرطوبة النسبية (%)", 
            0.0, 100.0, 50.0, 1.0,
            help="الرطوبة تؤثر على وضوح الغلاف الجوي"
        )
    
    with col2:
        st.markdown("**🌤️ الضغط والهطول**")
        mean_sea_level_pressure_MSL = st.slider(
            "ضغط مستوى البحر (hPa)", 
            950.0, 1050.0, 1013.0, 0.1,
            help="الضغط الجوي يؤثر على كثافة الهواء والإشعاع"
        )
        
        total_precipitation_sfc = st.slider(
            "إجمالي الهطول (mm)", 
            0.0, 50.0, 0.0, 0.1,
            help="المطر والثلج يقللان من الإشعاع الواصل للألواح"
        )
    
    with col3:
        st.markdown("**💨 سرعة الرياح**")
        wind_speed_10_m_above_gnd = st.slider(
            "سرعة الرياح على ارتفاع 10م (m/s)", 
            0.0, 25.0, 3.0, 0.1,
            help="الرياح تساعد في تبريد الألواح وتحسين الكفاءة"
        )
        
        st.markdown("**☀️ الإشعاع الشمسي**")
        shortwave_radiation_backwards_sfc = st.slider(
            "الإشعاع قصير الموجة (W/m²)", 
            0.0, 1400.0, 800.0, 1.0,
            help="كمية الإشعاع الشمسي المتاحة للتحويل إلى كهرباء"
        )
    
    st.markdown("---")
    st.subheader("☁️ الغطاء السحابي")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cloud_cover_sfc = st.slider(
            "الغطاء السحابي الإجمالي (%)", 
            0.0, 100.0, 30.0, 1.0,
            help="النسبة الإجمالية للسماء المغطاة بالسحب"
        )
    
    with col2:
        high_cloud_cover_high_cld_lay = st.slider(
            "الغطاء السحابي العالي (%)", 
            0.0, 100.0, 20.0, 1.0,
            help="السحب العالية الارتفاع"
        )
    
    with col3:
        low_cloud_cover_low_cld_lay = st.slider(
            "الغطاء السحابي المنخفض (%)", 
            0.0, 100.0, 15.0, 1.0,
            help="السحب المنخفضة الارتفاع"
        )
    
    st.markdown("---")
    st.subheader("🌅 الزوايا الشمسية")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        angle_of_incidence = st.slider(
            "زاوية السقوط (درجة)", 
            0.0, 90.0, 30.0, 0.5,
            help="الزاوية بين أشعة الشمس وسطح اللوح"
        )
    
    with col2:
        zenith = st.slider(
            "زاوية الذروة (درجة)", 
            0.0, 90.0, 45.0, 0.5,
            help="الزاوية بين الشمس وأعلى نقطة في السماء"
        )
    
    with col3:
        azimuth = st.slider(
            "زاوية السمت (درجة)", 
            0.0, 360.0, 180.0, 1.0,
            help="اتجاه الشمس من الشمال (0°=شمال، 180°=جنوب)"
        )
    
    # حساب المتغيرات المشتقة تلقائياً
    delta_angle = abs(angle_of_incidence - zenith)
    temp_humidity_index = temperature_2_m_above_gnd * relative_humidity_2_m_above_gnd
    
    st.markdown("---")
    st.subheader("🧮 المتغيرات المحسوبة تلقائياً")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**فرق الزاوية:** {delta_angle:.2f}° \n\n(الفرق بين زاوية السقوط وزاوية الذروة)")
    
    with col2:
        st.info(f"**مؤشر الحرارة-الرطوبة:** {temp_humidity_index:.2f} \n\n(درجة الحرارة × الرطوبة النسبية)")
    
    st.markdown("---")
    
    # زر التنبؤ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 تنبؤ إنتاج الطاقة الشمسية", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # إعداد البيانات للتنبؤ
            input_data = np.array([[
                temperature_2_m_above_gnd,
                relative_humidity_2_m_above_gnd,
                mean_sea_level_pressure_MSL,
                total_precipitation_sfc,
                total_cloud_cover_sfc,
                high_cloud_cover_high_cld_lay,
                low_cloud_cover_low_cld_lay,
                shortwave_radiation_backwards_sfc,
                wind_speed_10_m_above_gnd,
                angle_of_incidence,
                zenith,
                azimuth,
                delta_angle,
                temp_humidity_index
            ]])
            
            # تطبيق التقييس
            input_scaled = scaler.transform(input_data)
            
            # التنبؤ
            prediction = model.predict(input_scaled, verbose=0)[0][0]
            
            st.success("✅ تم التنبؤ بنجاح!")
            st.markdown("---")
            
            # عرض النتائج
            st.subheader("📊 نتائج التنبؤ")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("⚡ الإنتاج المتوقع", f"{prediction:.2f} kW", 
                         help="الطاقة المتوقع إنتاجها في الساعة")
            
            with col2:
                daily_production = prediction * 8
                st.metric("📅 الإنتاج اليومي", f"{daily_production:.1f} kWh",
                         help="الطاقة المتوقعة لمدة 8 ساعات إشعاع")
            
            with col3:
                monthly_production = daily_production * 30
                st.metric("📊 الإنتاج الشهري", f"{monthly_production:.0f} kWh",
                         help="الطاقة المتوقعة شهرياً")
            
            with col4:
                efficiency = min((prediction / 1000) * 100, 100) if prediction > 0 else 0
                st.metric("📈 الكفاءة المقدرة", f"{efficiency:.1f}%",
                         help="الكفاءة النسبية للنظام")
            
            # مؤشر دائري للإنتاج
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "إنتاج الطاقة المتوقع (kW)", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 1000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "orange"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 200], 'color': 'lightgray'},
                        {'range': [200, 400], 'color': 'yellow'},
                        {'range': [400, 600], 'color': 'orange'},
                        {'range': [600, 800], 'color': 'lightgreen'},
                        {'range': [800, 1000], 'color': 'green'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 750
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # تقييم الظروف
            st.subheader("📋 تقييم الظروف البيئية")
            
            conditions = []
            
            # تقييم درجة الحرارة
            if temperature_2_m_above_gnd > 35:
                conditions.append("🔥 درجة الحرارة عالية - قد تقلل الكفاءة")
            elif temperature_2_m_above_gnd < 0:
                conditions.append("❄️ درجة الحرارة منخفضة - قد تؤثر على الأداء")
            else:
                conditions.append("✅ درجة الحرارة مثلى")
            
            # تقييم الغطاء السحابي
            if total_cloud_cover_sfc > 80:
                conditions.append("☁️ غطاء سحابي كثيف - سيقلل الإنتاج بشكل كبير")
            elif total_cloud_cover_sfc > 50:
                conditions.append("🌤️ غطاء سحابي متوسط - قد يقلل الإنتاج")
            else:
                conditions.append("☀️ سماء صافية - ظروف ممتازة للإنتاج")
            
            # تقييم الإشعاع
            if shortwave_radiation_backwards_sfc > 1000:
                conditions.append("🌟 إشعاع شمسي عالي - ظروف ممتازة")
            elif shortwave_radiation_backwards_sfc < 200:
                conditions.append("🌫️ إشعاع شمسي منخفض - إنتاج محدود")
            else:
                conditions.append("☀️ إشعاع شمسي جيد")
            
            # تقييم زاوية السقوط
            if angle_of_incidence < 30:
                conditions.append("📐 زاوية سقوط مثلى - كفاءة عالية")
            elif angle_of_incidence > 60:
                conditions.append("📐 زاوية سقوط غير مثلى - كفاءة منخفضة")
            else:
                conditions.append("📐 زاوية سقوط جيدة")
            
            for condition in conditions:
                st.write(f"• {condition}")
            
        except Exception as e:
            st.error(f"❌ خطأ في التنبؤ: {e}")
            st.write("تأكد من صحة البيانات المدخلة")

else:
    st.error("❌ النموذج غير متاح. تأكد من وجود ملفات النموذج في المسار الصحيح.")
    st.info("الملفات المطلوبة: my_model.keras و scaler.pkl")

# الهامش السفلي
st.markdown("---")
st.markdown("**🌞 نظام التنبؤ بإنتاج الطاقة الشمسية** | مدعوم بالذكاء الاصطناعي 🤖")
