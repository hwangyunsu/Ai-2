#분류 결과 + 이미지 + 영상 + 텍스트 보여주기
#파일 이름 streamlit_app.py

import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1Yd9ooq-ZZ8uzOeH4nTTqlpcR1eOEsP7r'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'1Eb4QqDKYjeaX66xBoMEUNO3RzuTP8sTm'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_column_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(labels):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.image(f"https://i.ibb.co/Ny2q8tc/18014467256496901.jpg?text={label}", caption=f"이미지: {label}", use_column_width=True)

    # 2nd Row - YouTube Videos based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.video("https://www.youtube.com/watch?v=5JcdG5EbgYw", start_time=0)
            st.caption(f"유튜브: {label}")

    # 3rd Row - Text based on labels
    for i, label in enumerate(labels[:3]):
        with cols[i]:
            st.write(f"{label}....")

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 레이아웃 설정
left_column, right_column = st.columns(2)

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        display_right_content(labels)
