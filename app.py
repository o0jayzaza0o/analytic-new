import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import json_repair
from bs4 import BeautifulSoup
from unsloth import FastLanguageModel

# --- 1. ตั้งค่าและโหลดโมเดลจากโฟลเดอร์ที่ Save ไว้ ---
st.set_page_config(page_title="AI XML Analyst", layout="wide")

@st.cache_resource
def load_local_model():
    # โหลดจากโฟลเดอร์ที่เรา Save ไว้ (mysaved_model)
    # หมายเหตุ: ต้องแน่ใจว่าโฟลเดอร์ mysaved_model อยู่ที่เดียวกับ app.py
    model_path = "mysaved_model" 
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path, # ชี้ไปที่โฟลเดอร์
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        # เผื่อหาไฟล์ไม่เจอ ให้โหลดจากเน็ตแทน (Fallback)
        st.warning(f"หาโฟลเดอร์โมเดลไม่เจอ ({e}) กำลังโหลดจาก Unsloth แทน...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Llama-3.2-3B-Instruct",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

# --- 2. ฟังก์ชันแกะ XML Text และวิเคราะห์ ---
def process_xml_text(xml_string, model, tokenizer):
    # ใช้ BeautifulSoup แกะ XML string
    soup = BeautifulSoup(xml_string, 'xml')
    items = soup.find_all('item')
    
    if not items:
        # เผื่อกรณี user วางมาแค่ text ธรรมดา ไม่มี tag item
        # ให้ลองหาจาก root หรือถือว่าเป็น item เดียว
        if soup.find('title'):
            items = [soup]
        else:
            return []

    results = []
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(items)
    
    for i, item in enumerate(items):
        # ดึงข้อมูลจาก Tag
        title = item.find('title').get_text() if item.find('title') else "No Title"
        description = item.find('description').get_text() if item.find('description') else ""
        link = item.find('link').get_text() if item.find('link') else ""
        
        # รวมข้อความเพื่อส่งให้ AI
        full_text = f"Title: {title}\nDescription: {description}"
        input_text = full_text[:1500]
        
        status_text.text(f"⏳ กำลังวิเคราะห์: {title[:30]}...")
        
        # --- AI Inference Part ---
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a News Analyst. Extract public figures and sentiment.
        Output JSON only: {{"persons": ["Name1", "Name2"], "sentiment": "Positive/Negative/Neutral"}}
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        News: {input_text}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            use_cache=True,
            temperature=0.1
        )
        
        response = tokenizer.batch_decode(outputs)[0].split("assistant")[-1].strip()
        
        # Parse JSON
        try:
            data = json_repair.loads(response)
            sentiment = str(data.get('sentiment', 'Neutral'))
            persons = data.get('persons', [])
            if isinstance(persons, str): persons = [persons]
            persons = [str(p) for p in persons if isinstance(p, (str, int))]
        except:
            sentiment = "Error"
            persons = []
            
        results.append({
            "title": title,
            "sentiment_clean": sentiment,
            "persons_clean": persons,
            "link": link
        })
        
        progress_bar.progress((i + 1) / total)

    status_text.text("✅ วิเคราะห์เสร็จสิ้น!")
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 3. UI หน้าจอ ---
st.title("🤖 AI XML News Analyzer")
st.markdown("วางโค้ด XML (`<item>...</item>`) ลงในช่องด้านล่างเพื่อวิเคราะห์")

# โหลดโมเดล
with st.spinner("กำลังโหลดโมเดล..."):
    try:
        model, tokenizer = load_local_model()
        st.success("Model Loaded Successfully! 🚀")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Input Text Area (รับ XML)
xml_input = st.text_area("วาง XML Code ที่นี่:", height=300, placeholder="<item>\n<title>Example News</title>\n...</item>")

if st.button("🚀 เริ่มวิเคราะห์"):
    if not xml_input.strip():
        st.warning("กรุณาวางโค้ด XML ก่อนครับ")
    else:
        df = process_xml_text(xml_input, model, tokenizer)
        
        if not df.empty:
            st.session_state['data_xml'] = df
        else:
            st.error("ไม่พบข้อมูลใน XML หรือรูปแบบไม่ถูกต้อง")

# --- 4. แสดงผล ---
if 'data_xml' in st.session_state:
    df = st.session_state['data_xml']
    st.divider()
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนข่าว", len(df))
    c2.metric("ข่าวบวก", len(df[df['sentiment_clean']=='Positive']))
    c3.metric("ข่าวลบ", len(df[df['sentiment_clean']=='Negative']))
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Sentiment Analysis")
        fig_pie = px.pie(df, names='sentiment_clean', color='sentiment_clean',
                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#f1c40f'})
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Top Figures")
        all_persons = []
        for p_list in df['persons_clean']:
            all_persons.extend(p_list)
            
        if all_persons:
            from collections import Counter
            counts = Counter(all_persons).most_common(10)
            df_p = pd.DataFrame(counts, columns=['Name', 'Count'])
            st.plotly_chart(px.bar(df_p, x='Count', y='Name', orientation='h'), use_container_width=True)
            
    # Table
    st.subheader("ผลลัพธ์การวิเคราะห์")
    st.dataframe(df[['title', 'sentiment_clean', 'persons_clean']])
