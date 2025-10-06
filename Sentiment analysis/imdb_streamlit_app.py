import streamlit as st
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

# تحميل النموذج
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('imdb_sentiment_model.keras')
    return model

# تحميل قاموس الكلمات
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    return word_index

model = load_model()
word_index = load_word_index()

# عنوان التطبيق
st.title("🎬 تحليل مشاعر مراجعات الأفلام")
st.write("اكتب مراجعة فيلم وسيتم تحليل ما إذا كانت إيجابية أم سلبية")

# مربع النص
user_input = st.text_area("أدخل المراجعة هنا:", height=150)

if st.button("تحليل المشاعر"):
    if user_input:
        # تحويل النص إلى أرقام
        words = user_input.lower().split()
        sequence = []
        for word in words:
            if word in word_index:
                idx = word_index[word]
                if idx < 10000:  # نفس num_words المستخدم في التدريب
                    sequence.append(idx)
        
        if sequence:
            # Padding
            padded = pad_sequences([sequence], maxlen=467)
            
            # التنبؤ
            prediction = model.predict(padded)[0][0]
            
            # عرض النتيجة
            st.subheader("النتيجة:")
            if prediction > 0.5:
                st.success(f"✅ مراجعة إيجابية ({prediction*100:.1f}%)")
            else:
                st.error(f"❌ مراجعة سلبية ({(1-prediction)*100:.1f}%)")
            
            # شريط التقدم
            st.progress(float(prediction))
        else:
            st.warning("لم يتم التعرف على أي كلمات من المراجعة")
    else:
        st.warning("الرجاء إدخال نص المراجعة")
