import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import random
import cloudpickle

st.set_page_config(
    page_title="Hotel Booking System ",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    with open('model_with_cloud.pkl', 'rb') as f:
        return cloudpickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

with st.sidebar:
    st.title("Hotel Booking System")
    st.markdown("---")
    page = st.radio("Navigation", ["New Booking", "Test Cases"])
    st.markdown("---")
    st.info("This application predicts booking status based on input parameters.")

if page == "New Booking":
    st.title("Hotel Booking")
    st.markdown("Complete the form below to create a new booking and get a prediction.")
    
    with st.form("booking_form"):
        st.subheader("Booking Information")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            booking_id = st.text_input(
                "Booking ID",
                value=f"INN{random.randint(10000, 99999):05d}",
                help="Unique booking identifier"
            )
        
        with col2:
            no_of_adults = st.number_input("Number of Adults", min_value=0, value=1, step=1)
        
        with col3:
            no_of_children = st.number_input("Number of Children", min_value=0, value=0, step=1)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1, step=1)
        
        with col2:
            no_of_week_nights = st.number_input("Week Nights", min_value=0, value=2, step=1)
        
        with col3:
            meal_plans = ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"]
            type_of_meal_plan = st.selectbox("Type of Meal Plan", options=meal_plans)
        
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            room_types = ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"]
            room_type_reserved = st.selectbox("Room Type Reserved", options=room_types)
        
        with col2:
            lead_time = st.slider(
                "Lead Time (days)",
                min_value=0,
                max_value=365,
                value=30,
                help="Days between booking and arrival"
            )
        
        with col3:
            arrival_date_full = st.date_input("Arrival Date", value=date.today())
            arrival_year = arrival_date_full.year
            arrival_month = arrival_date_full.month
            arrival_date = arrival_date_full.day
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            market_segments = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
            market_segment_type = st.selectbox("Market Segment Type", options=market_segments)
        
        with col2:
            repeated_guest = st.radio(
                "Repeated Guest",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                horizontal=True
            )
        
        with col3:
            required_car_parking_space = st.radio(
                "Car Parking Required",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                horizontal=True
            )
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            no_of_previous_cancellations = st.number_input(
                "Previous Cancellations",
                min_value=0,
                value=0,
                step=1
            )
        
        with col2:
            no_of_previous_bookings = st.number_input(
                "Previous Bookings (Not Canceled)",
                min_value=0,
                value=0,
                step=1
            )
        
        with col3:
            no_of_special_requests = st.number_input(
                "Special Requests",
                min_value=0,
                value=0,
                step=1
            )
        
        with col4:
            avg_price_per_room = st.number_input(
                "Average Price Per Room ($)",
                min_value=0.0,
                value=100.0,
                step=5.0,
                format="%.2f"
            )
        
        submit_button = st.form_submit_button("Submit Booking", type="primary", use_container_width=True)
    
    if submit_button:
        with st.spinner("Processing booking..."):
            booking_data = {
                "Booking_ID": booking_id,
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
                "no_of_previous_bookings_not_canceled": no_of_previous_bookings,
                "no_of_special_requests": no_of_special_requests,
                "avg_price_per_room": avg_price_per_room
            }
            
            df = pd.DataFrame([booking_data])
            
            st.subheader("Booking Details")
            st.dataframe(df, use_container_width=True)
            
            if model_loaded:
                try:
                    _, labels = model.predict(df)
                    
                    st.subheader("Prediction Results")
                    if labels[0] == "Canceled":
                        st.error(f"This booking is predicted to be **{labels[0]}**")
                    else:
                        st.success(f"This booking is predicted to be **{labels[0]}**")
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.warning("Model not loaded. Cannot make prediction.")

elif page == "Test Cases":
    st.title("Test Cases")
    
    test_case_1_expander = st.expander("Test Case 1", expanded=True)
    with test_case_1_expander:
        
        test_case_1 = {
            "booking_id": "INN00101",
            "no_of_adults": 2,
            "no_of_children": 4,
            "no_of_weekend_nights": 4,
            "no_of_week_nights": 3,
            "type_of_meal_plan": "Meal Plan 3",
            "required_car_parking_space": 0,
            "room_type_reserved": "Room_Type 1",
            "lead_time": 225,
            "arrival_year": 2017,
            "arrival_month": 1,
            "arrival_date": 20,
            "market_segment_type": "Online",
            "repeated_guest": 1,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 0,
            "no_of_special_requests": 0,
            "avg_price_per_room": 105.75
        }
        
        df_test_1 = pd.DataFrame([test_case_1])
        st.dataframe(df_test_1, use_container_width=True)
        
        if st.button("Run Test Case 1"):
            with st.spinner("Making prediction..."):
                if model_loaded:
                    try:
                        _, labels = model.predict(df_test_1)
                        
                        if labels[0] == "Canceled":
                            st.error(f"Prediction: This booking is likely to be **{labels[0]}**")
                        else:
                            st.success(f"Prediction: This booking is likely to be **{labels[0]}**")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                else:
                    st.warning("Model not loaded. Cannot make prediction.")
    
    test_case_2_expander = st.expander("Test Case 2", expanded=True)
    with test_case_2_expander:
        test_case_2 = {
            "booking_id": "INN00202",
            "no_of_adults": 1,
            "no_of_children": 2,
            "no_of_weekend_nights": 2,
            "no_of_week_nights": 3,
            "type_of_meal_plan": "Meal Plan 2",
            "required_car_parking_space": 1,
            "room_type_reserved": "Room_Type 1",
            "lead_time": 12,
            "arrival_year": 2018,
            "arrival_month": 4,
            "arrival_date": 10,
            "market_segment_type": "Corporate",
            "repeated_guest": 1,
            "no_of_previous_cancellations": 2,
            "no_of_previous_bookings_not_canceled": 5,
            "no_of_special_requests": 2,
            "avg_price_per_room": 98.50
        }
        
        df_test_2 = pd.DataFrame([test_case_2])
        st.dataframe(df_test_2, use_container_width=True)
        
        if st.button("Run Test Case 2"):
            with st.spinner("Making prediction..."):
                if model_loaded:
                    try:
                        _, labels = model.predict(df_test_2)
                        if labels[0] == "Canceled":
                            st.error(f"Prediction: This booking is likely to be **{labels[0]}**")
                        else:
                            st.success(f"Prediction: This booking is likely to be **{labels[0]}**")
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                else:
                    st.warning("Model not loaded. Cannot make prediction.")