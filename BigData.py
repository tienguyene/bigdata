import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import stock_historical_data
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from scipy.optimize import minimize
import streamlit as st
from tensorflow.keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input

# Bước 1: Khởi tạo Spark Session
spark = SparkSession.builder \
    .appName("Vnstock Data") \
    .getOrCreate()

# Danh sách cổ phiếu cần phân tích
stock_list = ['VCB', 'FPT', 'ACV', 'HPG', 'VHM']
company_names = ["VCB", "FPT", "ACV", "HPG", "VHM"]

# Thiết lập thời gian bắt đầu và kết thúc
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Hàm lấy dữ liệu và xử lý lỗi
def get_stock_data(symbol):
    try:
        data = stock_historical_data(symbol, '2019-01-01', end.strftime('%Y-%m-%d'))
        return data
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu có lỗi

# Tạo một hàm để tính toán và hiển thị biểu đồ
def display_correlation_and_mean_return_risk():
    # Lấy dữ liệu VN-Index
    vnindex_data = stock_historical_data("VNINDEX", '2019-01-01', end.strftime('%Y-%m-%d'), "1D", "index")
    if not vnindex_data.empty:
        vnindex_data['company_name'] = 'VNINDEX'
        vnindex_spark_df = spark.createDataFrame(vnindex_data)  # Chuyển đổi sang DataFrame Spark
    else:
        vnindex_spark_df = None

    # Tạo DataFrame Spark cho từng cổ phiếu
    spark_dataframes = {}
    for stock, com_name in zip(stock_list, company_names):
        try:
            df = get_stock_data(stock)
            if not df.empty:
                df['company_name'] = com_name
                spark_df = spark.createDataFrame(df)
                spark_dataframes[stock] = spark_df
        except Exception as e:
            print(f"Error retrieving data for {stock}: {e}")

    # Tính toán log return cho từng cổ phiếu
    final_df = None
    for stock in stock_list:
        if stock in spark_dataframes:
            if final_df is None:
                final_df = spark_dataframes[stock].select("time", "close").withColumnRenamed("close", stock)
            else:
                final_df = final_df.join(spark_dataframes[stock].select("time", "close").withColumnRenamed("close", stock), on="time", how="outer")

    window_spec = Window.orderBy("time")
    final_df_with_lag = final_df
    for column in final_df.columns[1:]:
        final_df_with_lag = final_df_with_lag.withColumn(f"previous_{column}", F.lag(column).over(window_spec))

    log_return_exprs = [
        (F.log(F.col(column) / F.col(f"previous_{column}")).alias(f"log_return_{column}"))
        for column in final_df.columns[1:]
    ]
    log_return_df = final_df_with_lag.select("time", *log_return_exprs).na.drop()

    # Chọn các cột log return và chuyển đổi thành RDD
    log_return_columns = [f"log_return_{column}" for column in final_df.columns[1:]]
    log_return_rdd = log_return_df.select(log_return_columns).rdd.map(lambda row: Vectors.dense(row))

    # Tính ma trận tương quan và vẽ heatmap
    correlation_matrix = Statistics.corr(log_return_rdd, method="pearson")
    correlation_df = pd.DataFrame(correlation_matrix, columns=log_return_columns, index=log_return_columns)

    st.subheader("Correlation Matrix Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, fmt=".2f", cmap='summer', square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix Heatmap")
    st.pyplot(plt)

    # Tính toán hiệp phương sai
    row_matrix = RowMatrix(log_return_rdd)
    covariance_matrix = row_matrix.computeCovariance()
    covariance_matrix_scaled = np.array(covariance_matrix.toArray()) * 252
    covariance_df = pd.DataFrame(covariance_matrix_scaled, columns=log_return_columns, index=log_return_columns)

    # Tính mean và std cho từng cổ phiếu
    mean_std_df = log_return_df.select([F.mean(col).alias(f'mean_{col}') for col in log_return_df.columns] +
                                        [F.stddev(col).alias(f'std_{col}') for col in log_return_df.columns])
    mean_std_values = mean_std_df.collect()[0]
    mean_values = mean_std_values[:len(log_return_df.columns)]
    std_values = mean_std_values[len(log_return_df.columns):]

    mean_std_pandas_df = pd.DataFrame({
        'Mean Return': mean_values,
        'Standard Deviation': std_values
    }, index=log_return_df.columns)

    # Biểu đồ Mean Return vs. Risk
    st.subheader("Mean Return vs. Risk (Standard Deviation)")
    area = np.pi * 20
    plt.figure(figsize=(10, 8))
    plt.scatter(mean_std_pandas_df['Mean Return'], mean_std_pandas_df['Standard Deviation'], s=area)

    for label, x, y in zip(mean_std_pandas_df.index, mean_std_pandas_df['Mean Return'], mean_std_pandas_df['Standard Deviation']):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points',
                     ha='right', va='bottom',
                     arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

    plt.xlabel('Expected Return')
    plt.ylabel('Risk (Standard Deviation)')
    plt.title('Mean Return vs. Risk')
    plt.grid()
    st.pyplot(plt)

    # Tối ưu hóa danh mục đầu tư
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    risk_free_rate = 0.05
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.4) for _ in range(len(stock_list))]
    initial_weights = np.array([1 / len(stock_list)] * len(stock_list))

    # Chuyển đổi Spark DataFrame log_return thành Pandas DataFrame để tính toán optimize
    log_return_pandas_df = log_return_df.toPandas()
    log_return_pandas_df.set_index('time', inplace=True)

    cov_matrix = covariance_df.drop(covariance_df.columns[-1], axis=1)
    cov_matrix = cov_matrix.drop(cov_matrix.index[-1])

    optimized_results = minimize(lambda w: -sharpe_ratio(w, log_return_pandas_df, cov_matrix, risk_free_rate), 
                                  initial_weights, 
                                  method='SLSQP', 
                                  constraints=constraints, 
                                  bounds=bounds)

    optimal_weights = optimized_results.x

    st.write("Optimal Weights:")
    for ticker, weight in zip(stock_list, optimal_weights):
        st.write(f"{ticker}: {weight:.4f}")

    optimal_portfolio_return = expected_return(optimal_weights, log_return_pandas_df)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_return_pandas_df, cov_matrix, risk_free_rate)

    st.write(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
    st.write(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
    st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

    # Vẽ biểu đồ hình tròn cho tỷ lệ tối ưu
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(optimal_weights, labels=stock_list, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=12)
    plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.subheader("Optimal Portfolio Weights")
    st.pyplot(plt)

# Bước 2: Tải mô hình LSTM
def load_lstm_model():
    model = load_model("path_to_your_saved_lstm_model.h5")  # Đường dẫn tới mô hình LSTM đã lưu
    return model

# Dự đoán giá cổ phiếu cho các mã cổ phiếu
def predict_stock_price(stock_code):
    model = load_lstm_model()
    # Lấy dữ liệu của cổ phiếu
    stock_data = get_stock_data(stock_code)
    if stock_data.empty:
        return None

    stock_data['Date'] = pd.to_datetime(stock_data['time'])
    stock_data.set_index('Date', inplace=True)
    stock_data = stock_data[['close']]  # Giữ lại cột giá đóng cửa
    stock_data = stock_data.values  # Chuyển đổi thành mảng NumPy

    # Tiền xử lý dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    # Tạo dữ liệu đầu vào cho mô hình
    x_input = []
    for i in range(len(scaled_data) - 60):  # Sử dụng 60 giá trị trước để dự đoán
        x_input.append(scaled_data[i:i+60, 0])
    x_input = np.array(x_input)
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

    # Dự đoán giá
    predicted_price = model.predict(x_input)
    predicted_price = scaler.inverse_transform(predicted_price)  # Chuyển đổi lại giá về giá trị thực

    return predicted_price

# Bước 3: Tạo ứng dụng Streamlit
def main():
    st.title("Stock Price Analysis and Prediction")

    # Phân tích danh mục đầu tư
    st.header("Portfolio Analysis")
    display_correlation_and_mean_return_risk()

    # Dự đoán giá cổ phiếu
    st.header("Stock Price Prediction")
    stock_code = st.selectbox("Select Stock Code", stock_list)

    if st.button("Predict"):
        predicted_prices = predict_stock_price(stock_code)
        if predicted_prices is not None:
            st.line_chart(predicted_prices.flatten())
        else:
            st.write("Error retrieving stock data for prediction.")

if __name__ == "__main__":
    main()
