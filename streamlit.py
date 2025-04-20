import streamlit as st
import pandas as pd
import cloudpickle

st.title("Hotel Booking Prediction")
st.write("Masukkan detail pemesanan hotel untuk memprediksi apakah akan dibatalkan atau tidak.")

with open('model_with_cloud.pkl', 'rb') as f:
    model = cloudpickle.load(f)

col1, col2 = st.columns(2)
with col1:
    no_of_adults = st.selectbox("Jumlah Dewasa", [0, 1, 2, 3, 4])
    no_of_children = st.selectbox("Jumlah Anak", [0, 1, 2, 3, 9, 10])
    no_of_weekend_nights = st.selectbox("Malam Akhir Pekan", [0, 1, 2, 3, 4, 5, 6, 7])
    no_of_week_nights = st.selectbox("Malam Hari Kerja", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    type_of_meal_plan = st.selectbox("Paket Makanan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox("Butuh Parkir?", [0, 1])
with col2:
    room_type_reserved = st.selectbox("Tipe Kamar", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.slider("Lead Time (hari)", 0, 400, 30)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
    arrival_month = st.selectbox("Bulan Kedatangan", list(range(1, 13)))
    arrival_date = st.selectbox("Tanggal Kedatangan", list(range(1, 32)))
    market_segment_type = st.selectbox("Segmentasi Pasar", ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])

repeated_guest = st.selectbox("Pernah Menginap Sebelumnya?", [0, 1])
no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, step=1)
no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya yang Tidak Dibatalkan", min_value=0, step=1)
avg_price_per_room = st.number_input("Harga Rata-rata/Kamar", min_value=0.0, step=1.0)
no_of_special_requests = st.selectbox("Jumlah Permintaan Khusus", [0, 1, 2, 3, 4, 5])

if st.button("Prediksi"):
    input_data = pd.DataFrame([{
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests,
    }])

    prediction = model.predict(input_data)[0]
    label = "Canceled" if prediction == "Canceled" else "Not Canceled"
    st.subheader("Hasil Prediksi:")
    st.success(f"Hasil prediksi: **{label}**")

if st.button("Test Case 1"):
    st.session_state.update({
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 2,
        "no_of_week_nights": 3,
        "type_of_meal_plan": 'Meal Plan 1',
        "required_car_parking_space": 1,
        "room_type_reserved": 'Room_Type 1',
        "lead_time": 120,
        "arrival_year": 2018,
        "arrival_month": 8,
        "arrival_date": 15,
        "market_segment_type": 'Online',
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 5,
        "avg_price_per_room": 100.0,
        "no_of_special_requests": 1,
    })

if st.button("Test Case 2"):
    st.session_state.update({
        "no_of_adults": 1,
        "no_of_children": 2,
        "no_of_weekend_nights": 0,
        "no_of_week_nights": 1,
        "type_of_meal_plan": 'Not Selected',
        "required_car_parking_space": 0,
        "room_type_reserved": 'Room_Type 4',
        "lead_time": 300,
        "arrival_year": 2017,
        "arrival_month": 5,
        "arrival_date": 4,
        "market_segment_type": 'Offline',
        "repeated_guest": 1,
        "no_of_previous_cancellations": 3,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 75.5,
        "no_of_special_requests": 0,
    })
