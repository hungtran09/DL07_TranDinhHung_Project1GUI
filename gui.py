import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from TextPreprocess import text_process_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

#1. Load data
df_san_pham = pd.read_csv('San_pham.csv')
df_khach_hang = pd.read_csv('Khach_hang.csv')
df_danh_gia = pd.read_csv('Preprocess_San_pham.csv').dropna()


#2. Function process data:
def process_customer_and_product_data(df_danh_gia, df_san_pham, df_khach_hang=None):
    df_danh_gia_2 = df_danh_gia.merge(df_san_pham[['ma_san_pham', 'ten_san_pham']], on='ma_san_pham', how='left')
    if df_khach_hang is not None:
        df_danh_gia_2 = df_danh_gia_2.merge(df_khach_hang, on='ma_khach_hang', how='left')
    
    customer_product_counts = (
        df_danh_gia_2.groupby(['ma_khach_hang', 'ho_ten', 'ten_san_pham'])
        .size()
        .reset_index(name='so_luong_mua')
        .sort_values(by='so_luong_mua', ascending=False)
    )
    
    product_purchase_counts = (
        df_danh_gia_2.groupby(['ma_san_pham', 'ten_san_pham'])
        .size()
        .reset_index(name='so_lan_mua')
        .sort_values(by='so_lan_mua', ascending=False)
    )
    
    return customer_product_counts, product_purchase_counts

# Lấy dữ liệu phân tích
customer_product_counts, product_purchase_counts = process_customer_and_product_data(df_danh_gia, df_san_pham, df_khach_hang)


#3. Load models 
# Đọc model
pkl_filename = "RandomForestModel.pkl" 
with open(pkl_filename, 'rb') as file:  
    rf_model = pickle.load(file)
# doc model tf idf vectorizer len
pkl_vectorize_filename = "tfidf_model.pkl"  
with open(pkl_vectorize_filename, 'rb') as file:  
    vectorizer_model = pickle.load(file)

#--------------
# GUI
st.title("Sentiment Analysis Project")

menu = ["Business Objective", "New Prediction", "Product analysis", "Customer analysis"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Trần Đình Hùng
                 Phạm Thị Ngọc Huyền""")
st.sidebar.write("""#### Giảng viên hướng dẫn: 
                    Khuất Thuỳ Phương""")
st.sidebar.write("""#### Thời gian báo cáo: 15/12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### HASAKI.VN là hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp  chuyên sâu với hệ thống cửa hàng trải dài trên  toàn quốc; và hiện đang là đối tác phân phối  chiến lược tại thị trường Việt Nam của hàng  loạt thương hiệu lớn...
    
    ###### Khách hàng có thể lên đây để lựa chọn sản phẩm, xem các đánh giá/nhận xét cũng như đặt mua sản phẩm.

    """)  
    st.write("""###### => Problem/ Requirement: Từ những đánh giá của khách hàng, làm thế nào để hiểu rõ cảm nhận của kháchh hàng về sản phẩm để cải thiện chất lượng sản phẩm và các dịch vụ đi kèm.""")
    st.image("Hasaki0.jpg")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    original_lines = None  # Biến lưu trữ dữ liệu gốc
    processed_lines = None  # Biến lưu trữ dữ liệu đã xử lý

    # Lựa chọn kiểu nhập liệu: Upload file hoặc Nhập thủ công
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))

    if type == "Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            try:
                original_lines = pd.read_csv(uploaded_file_1, header=None, encoding='utf8')[0]
                flag = True
            except Exception as e:
                st.error(f"Error reading the uploaded file: {e}")

    if type == "Input":
        # Nhập liệu thủ công
        content = st.text_area(label="Input your content:")
        if content.strip() != "":
            original_lines = np.array([content])
            flag = True

    # Xử lý khi có dữ liệu đầu vào
    if flag:
        try:
            # Process the data for prediction
            processed_lines = text_process_pipeline(original_lines)
            x_new = vectorizer_model.transform(processed_lines)
            y_pred_new = rf_model.predict(x_new)

            # Create a DataFrame for results
            results_df = pd.DataFrame({
                "Input": original_lines,  # Dữ liệu gốc
                "Label": y_pred_new       # Kết quả dự đoán
            })

            # Map numerical labels to human-readable text
            label_mapping = {0: "Bad", 1: "Neutral", 2: "Good"}
            results_df["Label"] = results_df["Label"].map(label_mapping)

            # Display the results DataFrame
            st.write("Prediction Results:")
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
    else:
        # Hiển thị thông báo lỗi nếu không có đầu vào
        st.warning("No data provided. Please upload a file or input content.")
            
elif choice == 'Product analysis':
    st.subheader("Product analysis")
    st.write("##### Choose product:")
    ma_san_pham_drop_down_box = [422204259, 205100137, 422211447]

    ma_san_pham_choice = st.selectbox('Mã sản phẩm', ma_san_pham_drop_down_box)

    if st.button("Phân tích sản phẩm"):
        ma_san_pham = int(ma_san_pham_choice)
        product_info = df_san_pham[df_san_pham['ma_san_pham'] == ma_san_pham]

        if product_info.empty:
                st.write(f"Không tìm thấy sản phẩm với mã sản phẩm: {ma_san_pham}")
        else:
            product_name = product_info['ten_san_pham'].values[0]
            st.write(f"**Phân tích cho sản phẩm:** {product_name} (Mã: {ma_san_pham})")

            # Lọc nhận xét cho sản phẩm
            product_reviews = df_danh_gia[df_danh_gia['ma_san_pham'] == ma_san_pham]

            # Tổng số nhận xét
            total_reviews = product_reviews.shape[0]
            st.write(f"**Tổng số nhận xét:** {total_reviews}")

            # Số nhận xét tích cực, tiêu cực, trung lập
            label_counts = product_reviews['label'].value_counts()
            st.write(f"**Số nhận xét tích cực:** {label_counts.get('good', 0)}")
            st.write(f"**Số nhận xét tiêu cực:** {label_counts.get('bad', 0)}")
            st.write(f"**Số nhận xét trung lập:** {label_counts.get('neutral', 0)}")

            # Trích xuất từ khóa chính
            st.write("### Word Cloud:")
            all_text = ' '.join(product_reviews['noi_dung_binh_luan'])
            vectorizer = CountVectorizer(stop_words='english', max_features=20)
            keywords = vectorizer.fit_transform([all_text])
            keyword_counts = Counter(vectorizer.get_feature_names_out())

            # Tạo word cloud từ từ khóa
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(keyword_counts))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # Biểu đồ tỷ lệ nhận xét
            st.write("### Biểu đồ tỷ lệ nhận xét:")
            fig1, ax1 = plt.subplots()
            ax1.pie(
                [label_counts.get('good', 0), label_counts.get('bad', 0), label_counts.get('neutral', 0)],
                labels=['Tích cực', 'Tiêu cực', 'Trung lập'],
                autopct='%1.1f%%',
                colors=['green', 'red', 'gray']
            )
            ax1.set_title("Tỷ lệ nhận xét")
            st.pyplot(fig1)

            # Biểu đồ số nhận xét theo thời gian
            if 'ngay_binh_luan' in product_reviews.columns:
                product_reviews['date'] = pd.to_datetime(product_reviews['ngay_binh_luan'], errors='coerce', format='%d/%m/%Y')
                reviews_over_time = product_reviews.groupby(product_reviews['date'].dt.to_period('M')).size()
                st.write("### Số nhận xét theo thời gian:")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                reviews_over_time.plot(kind='line', marker='o', ax=ax2, color='blue')
                ax2.set_title("Số nhận xét theo thời gian")
                ax2.set_xlabel("Thời gian")
                ax2.set_ylabel("Số nhận xét")
                ax2.grid(True)
                st.pyplot(fig2)
            else:
                st.write("Không có dữ liệu ngày bình luận để vẽ biểu đồ theo thời gian.")

            # Biểu đồ số sao trung bình theo thời gian
            if 'so_sao' in product_reviews.columns and 'ngay_binh_luan' in product_reviews.columns:
                avg_stars_over_time = product_reviews.groupby(product_reviews['date'].dt.to_period('M'))['so_sao'].mean()
                st.write("### Số sao trung bình theo thời gian:")
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                avg_stars_over_time.plot(kind='line', marker='o', ax=ax3, color='orange')
                ax3.set_title("Số sao trung bình theo thời gian")
                ax3.set_xlabel("Thời gian")
                ax3.set_ylabel("Số sao trung bình")
                ax3.grid(True)
                st.pyplot(fig3)
            else:
                st.write("Không có dữ liệu số sao hoặc ngày bình luận để vẽ biểu đồ số sao trung bình theo thời gian.")    



elif choice == 'Customer analysis':
    st.subheader("Customer analysis")
    st.write("##### Choose customer ID:")
    ma_khach_hang_drop_down_box = [121, 365, 833, 4735, 399]
    ma_khach_hang_choice = st.selectbox('Mã khách hàng', ma_khach_hang_drop_down_box)

    if st.button("Phân tích khách hàng"):
        customer_info = df_khach_hang[df_khach_hang['ma_khach_hang'] == ma_khach_hang_choice]

        if customer_info.empty:
            st.write(f"Không tìm thấy thông tin khách hàng với mã: {ma_khach_hang_choice}")
        else:
            customer_name = customer_info['ho_ten'].values[0]
            st.write(f"**Phân tích cho khách hàng:** {customer_name} (Mã: {ma_khach_hang_choice})")

            # Sản phẩm đã mua
            purchased_products = customer_product_counts[
                customer_product_counts['ma_khach_hang'] == ma_khach_hang_choice
            ]
            st.write("### Sản phẩm đã mua:")
            st.dataframe(purchased_products[['ten_san_pham', 'so_luong_mua']])
            
            # Nhận xét của khách hàng
            customer_reviews = df_danh_gia[df_danh_gia['ma_khach_hang'] == ma_khach_hang_choice]
            st.write("### Nhận xét của khách hàng:")
            st.dataframe(customer_reviews[['noi_dung_binh_luan', 'label']])

            # Tỷ lệ nhận xét của khách hàng
            st.write("### Tỷ lệ nhận xét của khách hàng:")
            label_counts = customer_reviews['label'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(
                [label_counts.get('good', 0), label_counts.get('bad', 0), label_counts.get('neutral', 0)],
                labels=['Tích cực', 'Tiêu cực', 'Trung lập'],
                autopct='%1.1f%%',
                colors=['green', 'red', 'gray']
            )
            ax.set_title("Tỷ lệ nhận xét")
            st.pyplot(fig)