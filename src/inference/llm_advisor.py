import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Bạn là chuyên gia nông nghiệp. Dựa trên kết quả phân tích hình ảnh nông sản,
hãy đưa ra tư vấn ngắn gọn bằng tiếng Việt về:
1. Đánh giá chất lượng
2. Nguyên nhân có thể
3. Khuyến nghị xử lý (bảo quản / xuất bán / loại bỏ)
Trả lời trong 3-4 câu, ngắn gọn, dễ hiểu."""

def get_advice(clf_result: dict, det_result: dict) -> str:
    prompt = f"""
Kết quả phân tích nông sản:
- Loại: {clf_result['label']} (độ tin cậy: {clf_result['confidence']}%)
- Chất lượng: {clf_result['quality']}
- Số vùng khuyết tật: {det_result['num_defects']}
- Chi tiết: {[d['class'] for d in det_result['detections']]}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content