
# 🏨 Hotel Booking Cancellation Classification – UTS Deployment

## Student Info
- **Name:** Steven Liementha
- **NIM:** 2702265370
- **Class:** LA09

### Streamlit Demo
🔗 **Live App:** [https://uts-model-deployment-s7nfqq2xsjvcr3yqml22zx.streamlit.app](https://uts-model-deployment-s7nfqq2xsjvcr3yqml22zx.streamlit.app)  
*Note: This link may expire or become inactive over time.*

## ⚙️ Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- **Python version:** `3.11.11` (recommended)

---

## Case Study Overview

As a Data Scientist, you are tasked with building and deploying a machine learning model to classify hotel booking cancellations. The model should predict whether a booking will be **Canceled** or **Not Canceled** based on the features provided in the dataset.

---

## Dataset Description

The dataset includes the following features:

| Column | Description |
|--------|-------------|
| `Booking_ID` | Unique ID of the booking |
| `no_of_adults` | Number of adults |
| `no_of_children` | Number of children |
| `no_of_weekend_nights` | Weekend nights booked |
| `no_of_week_nights` | Weekday nights booked |
| `type_of_meal_plan` | Type of meal plan selected |
| `required_car_parking_space` | Parking required? (0/1) |
| `room_type_reserved` | Encrypted room type |
| `lead_time` | Days before arrival when booking was made |
| `arrival_year` | Year of arrival |
| `arrival_month` | Month of arrival |
| `arrival_date` | Date of arrival |
| `market_segment_type` | Market segment |
| `repeated_guest` | Is the guest a repeater? (0/1) |
| `no_of_previous_cancellations` | Previous cancellations |
| `no_of_previous_bookings_not_canceled` | Previous successful bookings |
| `avg_price_per_room` | Average daily room price (euro) |
| `no_of_special_requests` | Number of special requests |
| `booking_status` | **Target** - Booking canceled or not |

### Create and Activate Conda Environment

```bash
conda create --name mlops python=3.11

conda activate mlops

pip install -r requirements.txt

python model.py # for training model and export model

python inference.py # for prediction

streamlit run streamlit.py #to run streamlit locally
```

### Note : To Update requirements.txt
```sh
pip list --format=freeze > requirements.txt
```
