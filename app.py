import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
from fpdf import FPDF, XPos, YPos
import os

# Cấu hình trang Dashboard
st.set_page_config(page_title="Phân tích Dữ liệu Học tập LMS", layout="wide")

st.title("📊 Hệ thống Phân tích Hành vi LMS & Sinh Báo cáo Khoa học Tự động")
st.markdown("**Đề tài:** *Dự báo kết quả học tập và cảnh báo sớm sinh viên có nguy cơ rớt môn trên hệ thống LMS bằng quy trình tích hợp KNIME, Looker Studio và AI tạo sinh*")
st.caption("Nhóm thực hiện: Phạm Trọng Nguyễn & Cộng sự | Trường CNTT&TT – Đại học Cần Thơ")
st.divider()

# 1. Tải & Làm sạch dữ liệu
st.sidebar.header("1. Nguồn dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu LMS (tùy chọn)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists("data/lms_student_data.csv"):
        df = pd.read_csv("data/lms_student_data.csv")
    else:
        st.error("Không tìm thấy tệp data/lms_student_data.csv. Vui lòng tải file CSV lên ở thanh bên trái.")
        st.stop()

# Làm sạch dữ liệu cơ bản (loại bỏ trùng lặp, điền giá trị thiếu bằng trung vị)
df = df.drop_duplicates()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 2. Tùy chọn biến tương tác trên Sidebar
st.sidebar.header("2. Tùy chỉnh Biểu đồ & Báo cáo")
selected_x = st.sidebar.selectbox("Chọn biến hành vi (Trục X):", [c for c in numeric_cols if c != "Diem_cuoi_ky"], index=0)
selected_y = st.sidebar.selectbox("Chọn biến kết quả (Trục Y):", numeric_cols, index=len(numeric_cols)-1)

# 3. Huấn luyện mô hình Hồi quy & Phân loại
features = ["So_lan_dang_nhap", "Gio_tu_hoc", "Ty_le_nop_bai", "Diem_giua_ky"]
available_features = [f for f in features if f in df.columns]

X = df[available_features]
y_reg = df["Diem_cuoi_ky"]
y_clf = df["Trang_thai"]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.25, random_state=42
)

# Mô hình 1: Hồi quy tuyến tính dự báo điểm cuối kỳ
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
r2 = r2_score(y_reg_test, reg_model.predict(X_test))

# Mô hình 2: Cây quyết định phân loại nguy cơ rớt môn
clf_model = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_model.fit(X_train, y_clf_train)
acc = accuracy_score(y_clf_test, clf_model.predict(X_test))

# 4. Hiển thị chỉ số tổng quan trên Dashboard
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng số sinh viên", len(df))
col2.metric("Sinh viên nguy cơ rớt", len(df[df["Trang_thai"] == "Nguy_co_rot"]))
col3.metric("Hệ số R² (Hồi quy)", f"{r2:.2f}")
col4.metric("Độ chính xác Phân loại", f"{acc*100:.1f}%")

# 5. Khu vực Biểu đồ tương tác & Dữ liệu
tab1, tab2, tab3 = st.tabs(["📈 Trực quan hóa Tương tác", "📋 Dữ liệu & Cảnh báo sớm", "📄 Xuất Báo cáo PDF"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            df, x=selected_x, y=selected_y, color="Trang_thai",
            title=f"Tương quan giữa {selected_x} và {selected_y}",
            color_discrete_map={"Dat": "#1f77b4", "Nguy_co_rot": "#d62728"},
            hover_data=["Ma_SV"] if "Ma_SV" in df.columns else None
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    with c2:
        fig_box = px.box(
            df, x="Trang_thai", y=selected_x, color="Trang_thai",
            title=f"Phân phối {selected_x} theo Nhóm trạng thái",
            color_discrete_map={"Dat": "#1f77b4", "Nguy_co_rot": "#d62728"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("Danh sách sinh viên thuộc nhóm Cảnh báo sớm (Nguy cơ rớt môn)")
    risk_df = df[df["Trang_thai"] == "Nguy_co_rot"]
    st.dataframe(risk_df, use_container_width=True)
    with st.expander("Xem toàn bộ dữ liệu đã làm sạch"):
        st.dataframe(df, use_container_width=True)

# 6. Hàm tự động vẽ biểu đồ tĩnh & sinh báo cáo PDF chuẩn học thuật
def generate_scientific_pdf(dataframe, r2_val, acc_val, x_col, y_col):
    chart_file = "temp_report_chart.png"
    plt.figure(figsize=(7.5, 4))
    for status, color in [("Dat", "#1f4e78"), ("Nguy_co_rot", "#c00000")]:
        sub = dataframe[dataframe["Trang_thai"] == status]
        plt.scatter(sub[x_col], sub[y_col], label=status, color=color, s=60, alpha=0.85)
    plt.title(f"BIEU DO TUONG QUAN: {x_col} vs {y_col}", fontweight="bold")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(chart_file, dpi=300)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Tiêu đề
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "BAO CAO KHOA HOC: PHAN TICH DU LIEU HOC TAP LMS", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Nhom tac gia: Pham Trong Nguyen - Truong CNTT&TT, Dai hoc Can Tho", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)

    # 1. Giới thiệu đề tài
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Gioi thieu de tai", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, "De tai ung dung quy trinh tich hop KNIME, Looker Studio va AI tao sinh nham phan tich hanh vi tuong tac cua sinh vien tren he thong LMS, qua do du bao ket qua hoc tap va canh bao som nhom sinh vien co nguy co rot mon.")
    pdf.ln(3)

    # 2. Mô tả dữ liệu
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Mo ta du lieu", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    so_sv = len(dataframe)
    so_nguy_co = len(dataframe[dataframe["Trang_thai"] == "Nguy_co_rot"])
    pdf.multi_cell(0, 5.5, f"Bo du lieu gom {so_sv} ban ghi sinh vien duoc lam sach va chuan hoa. Cac dac trung hanh vi chinh bao gom: So lan dang nhap LMS, Gio tu hoc, Ty le nop bai dung han (%), Diem giua ky va Diem cuoi ky. Co {so_nguy_co}/{so_sv} sinh vien thuoc nhom canh bao nguy co rot mon.")
    pdf.ln(3)

    # 3. Kết quả phân tích
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Ket qua phan tich & Mo hinh du bao", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, f"- Mo hinh Hoi quy tuyen tinh (du bao diem cuoi ky): He so xac dinh R2 dat {r2_val:.2f}.\n- Mo hinh Cay quyet dinh (phan loai nguy co rot): Do chinh xac dat {acc_val*100:.1f}%.\n- Trung binh nhom Dat co ty le nop bai {dataframe[dataframe['Trang_thai']=='Dat']['Ty_le_nop_bai'].mean():.1f}%, cao hon nhieu so voi nhom Nguy co rot ({dataframe[dataframe['Trang_thai']=='Nguy_co_rot']['Ty_le_nop_bai'].mean():.1f}%).")
    pdf.ln(3)

    # 4. Biểu đồ minh họa
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "4. Bieu do minh hoa", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(chart_file, x=25, w=160)
    pdf.ln(3)

    # 5. Kết luận & Đề xuất
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "5. Ket luan & De xuat", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, "Ket luan: Tan suat dang nhap va ty le nop bai dung han la hai chi bao quan trong nhat phan anh ket qua hoc tap. De xuat: Giang vien can gui thong bao nhac nho tu dong qua LMS ngay tu tuan thu 5 doi voi cac sinh vien co ty le nop bai duoi 65%.")

    pdf_bytes = bytes(pdf.output())
    if os.path.exists(chart_file):
        os.remove(chart_file)
    return pdf_bytes

with tab3:
    st.subheader("Tự động sinh Báo cáo Khoa học định dạng PDF")
    st.write("Báo cáo sẽ tự động tổng hợp số liệu mới nhất, kết quả huấn luyện mô hình và biểu đồ tương quan bạn đang chọn.")
    pdf_data = generate_scientific_pdf(df, r2, acc, selected_x, selected_y)
    st.download_button(
        label="⬇️ Tải xuống Báo cáo Khoa học (.PDF)",
        data=pdf_data,
        file_name="Bao_cao_khoa_hoc_LMS.pdf",
        mime="application/pdf"
    )
