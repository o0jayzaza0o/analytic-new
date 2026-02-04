import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from collections import Counter

# --- ส่วนโหลดข้อมูล ---
@st.cache_data
def load_data():
    # อ่านไฟล์ CSV ที่เราทำเสร็จแล้ว
    return pd.read_csv('global_news_analysis.csv')

try:
    df = load_data()
except:
    st.error("ไม่พบไฟล์ CSV กรุณารันส่วนวิเคราะห์ข่าวให้เสร็จก่อน")
    st.stop()

# --- ส่วนแสดงผล Dashboard ---
st.title("🌍 AI News Analyst Dashboard")
st.write("วิเคราะห์ข่าวโดย: **Llama 3.2 (Unsloth)**")

# Metrics
col1, col2 = st.columns(2)
col1.metric("Total News", len(df))
col1.metric("Positive Sentiment", len(df[df['sentiment_clean']=='Positive']))

# Charts
st.subheader("Sentiment Overview")
fig = px.pie(df, names='sentiment_clean', title='Sentiment Distribution', 
             color='sentiment_clean',
             color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
st.plotly_chart(fig)

st.subheader("Top Entities")
# (ใส่ Logic ระเบิด List รายชื่อคนตรงนี้แบบย่อ)
all_persons = []
for p in df['persons_clean']:
    try:
        # แปลง string เป็น list ถ้าจำเป็น
        val = ast.literal_eval(p) if isinstance(p, str) else p
        if isinstance(val, list): all_persons.extend([str(x) for x in val])
    except: pass

if all_persons:
    counts = Counter(all_persons).most_common(10)
    df_p = pd.DataFrame(counts, columns=['Name', 'Count'])
    st.plotly_chart(px.bar(df_p, x='Count', y='Name', orientation='h'))