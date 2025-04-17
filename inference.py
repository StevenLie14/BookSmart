import pandas as pd
from model import HotelPredictionModel

model = HotelPredictionModel.load_model('model.pkl')

inference_data = pd.DataFrame([{
    'Booking_ID': 'INN06348',
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 2,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 2',
    'required_car_parking_space': 1,
    'room_type_reserved': 'Room_Type 3',
    'lead_time': 87,
    'arrival_year': 2018,
    'arrival_month': 7,
    'arrival_date': 18,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 1,
    'no_of_previous_bookings_not_canceled': 5,
    'avg_price_per_room': 132.50,
    'no_of_special_requests': 2,
}])


# print(model)
predictions,labels = model.predict(inference_data)
print(predictions,labels)