# Hệ thống Phân tích Dữ liệu Học tập LMS & Sinh Báo cáo Khoa học Tự động

Mini-project thuộc học phần Thực hành Công nghệ số & AI — Trường Công nghệ Thông tin và Truyền thông, Đại học Cần Thơ.

## Thành viên thực hiện
- **Phạm Trọng Nguyễn** (và nhóm nghiên cứu)

## Chức năng chính
1. **Làm sạch & Phân tích dữ liệu:** Xử lý dữ liệu hành vi học tập trên LMS (`pandas`, `numpy`), huấn luyện mô hình Hồi quy tuyến tính và Cây quyết định (`scikit-learn`).
2. **Dashboard tương tác (Streamlit):** Cho phép tải dữ liệu CSV, tùy chọn biến hành vi để vẽ biểu đồ tương tác (`plotly`) và lọc danh sách sinh viên cần cảnh báo sớm.
3. **Sinh báo cáo PDF tự động (`fpdf2`):** Xuất trực tiếp báo cáo khoa học gồm 5 phần (Giới thiệu, Mô tả dữ liệu, Kết quả phân tích, Biểu đồ minh họa, Kết luận & Đề xuất).

## Hướng dẫn cài đặt và chạy ứng dụng
```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 2. Khởi chạy Dashboard Streamlit
streamlit run app.py
