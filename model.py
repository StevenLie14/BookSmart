import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle

class HotelPredictionModel:
    def __init__(self):
        self.model =  None
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.ohe_encoders = {}
        self.categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.num_columns = ['lead_time', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'avg_price_per_room', 'no_of_previous_cancellations',
            'no_of_previous_bookings_not_canceled', 'no_of_special_requests','arrival_year']
        self.all_columns = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully with Shape {}".format(self.data.shape))
        return self.data

    def preprocess_data(self,test_size = 0.2):
        self.x_train, self.y_train, self.x_test, self.y_test = self.clean_data(test_size)
        self.x_train[self.num_columns] = self.scaler.fit_transform(self.x_train[self.num_columns])
        self.x_test[self.num_columns] = self.scaler.transform(self.x_test[self.num_columns])
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)
        self.one_hot_encode_column()
        self.x_train = self.cyclic_encoding(self.x_train)
        self.x_test = self.cyclic_encoding(self.x_test)
        self.all_columns = self.x_train.columns

    def cyclic_encoding(self,data):
        if 'arrival_date' in data.columns and 'arrival_month' in data.columns:
            data['arrival_date_sin'] = np.sin(2 * np.pi * data['arrival_date'] / 31)
            data['arrival_date_cos'] = np.cos(2 * np.pi * data['arrival_date'] / 31)
            
            data['arrival_month_sin'] = np.sin(2 * np.pi * data['arrival_month'] / 12)
            data['arrival_month_cos'] = np.cos(2 * np.pi * data['arrival_month'] / 12)

            data.drop('arrival_date', axis=1, inplace=True)
            data.drop('arrival_month', axis=1, inplace=True)
        else:
            print("Warning: arrival_date or arrival_month column not found in data.")
        return data
    
    def one_hot_encode_column(self):
        for col in self.categorical_columns:
            self.ohe_encoders[col] = OneHotEncoder(sparse_output=False)
            train_encoded = self.ohe_encoders[col].fit_transform(self.x_train[[col]])
            test_encoded = self.ohe_encoders[col].transform(self.x_test[[col]])

            col_names = [f'{col}_{cat}' for cat in self.ohe_encoders[col].categories_[0]]

            train_ohe_df = pd.DataFrame(train_encoded, columns=col_names, index=self.x_train.index)
            test_ohe_df = pd.DataFrame(test_encoded, columns=col_names, index=self.x_test.index)

            self.x_train = pd.concat([self.x_train.drop(columns=[col]), train_ohe_df], axis=1)
            self.x_test = pd.concat([self.x_test.drop(columns=[col]), test_ohe_df], axis=1)
        return
    
    def clean_data(self,test_size):
        if self.data is None:
            print("Data is not loaded. Please load the data first.")
            return
        self.data =self.data.drop(columns=["Booking_ID"])
        x = self.data.drop(columns=["booking_status"])
        y = self.data["booking_status"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        avg_price_per_room_imputation = x_train["avg_price_per_room"].median()
        required_car_parking_space_imputation = x_train["required_car_parking_space"].mode()[0]
        type_of_meal_plan_imputation = x_train["type_of_meal_plan"].mode()[0]

        x_train['avg_price_per_room'] = x_train['avg_price_per_room'].fillna(avg_price_per_room_imputation)
        x_test['avg_price_per_room'] = x_test['avg_price_per_room'].fillna(avg_price_per_room_imputation)

        x_train['required_car_parking_space'] = x_train['required_car_parking_space'].fillna(required_car_parking_space_imputation)
        x_test['required_car_parking_space'] = x_test['required_car_parking_space'].fillna(required_car_parking_space_imputation)

        x_train['type_of_meal_plan'] = x_train['type_of_meal_plan'].fillna(type_of_meal_plan_imputation)
        x_test['type_of_meal_plan'] = x_test['type_of_meal_plan'].fillna(type_of_meal_plan_imputation)
        return x_train, y_train, x_test, y_test

    def train_model(self):
        print("Training Model")
        self.model = XGBClassifier(learning_rate = 0.2, max_depth = 6, n_estimators = 100, random_state = 42)
        self.model.fit(self.x_train, self.y_train)
        print("Training Model Done")
        return self.model
    
    def evaluate_model(self):
        print("Evaluating Model")
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred,target_names=self.label_encoder.classes_))
        print("Confusion Matrix : ")
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        print("Evaluating Model Done")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save_model(self, filename = 'model.pkl',cloud = False):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'encoders': self.ohe_encoders,
            'categorical_columns': self.categorical_columns,
            'num_columns': self.num_columns,
            'all_columns' : self.x_train.columns
        }
        
        if cloud:
            with open(filename, 'wb') as f:
                cloudpickle.dump(self, f)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
        return
        
    
    @classmethod
    def load_model(cls, filename = 'model.pkl',cloud = False):
        if cloud:
            with open(filename, 'rb') as file:
                model = cloudpickle.load(file)
                return model
        else:
            with open(filename, 'rb') as file:
                model_data = pickle.load(file)
            
        predictor = cls()
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoder = model_data['label_encoder']
        predictor.ohe_encoders = model_data['encoders']
        predictor.categorical_columns = model_data['categorical_columns']
        predictor.num_columns = model_data['num_columns']
        predictor.all_columns = model_data['all_columns']
        return predictor
    
    def preprocess_predict_data(self,new_data):
        new_data = new_data.drop(columns=[col for col in ['Booking_ID', 'booking_status'] if col in new_data.columns])
        preprocess_data = self.cyclic_encoding(new_data)
        preprocess_data[self.num_columns] = self.scaler.transform(preprocess_data[self.num_columns])
        for col in self.categorical_columns:
            test_encoded = self.ohe_encoders[col].transform(preprocess_data[[col]])
            col_names = [f'{col}_{cat}' for cat in self.ohe_encoders[col].categories_[0]]
            test_ohe_df = pd.DataFrame(test_encoded, columns=col_names, index=preprocess_data.index)
            preprocess_data = pd.concat([preprocess_data.drop(columns=[col]), test_ohe_df], axis=1)
        preprocess_data = preprocess_data.reindex(columns=self.all_columns)
        if set(preprocess_data.columns) != set(self.all_columns):
            missing = set(self.all_columns) - set(preprocess_data.columns)
            extra = set(preprocess_data.columns) - set(self.all_columns)
            print("Mismatch in test data columns.")
            if missing:
                print(f"Missing columns: {missing}")
            if extra:
                print(f"Unexpected columns: {extra}")
            return None
        
        return preprocess_data
    
    def predict(self, test_data):
        preprocess_data = self.preprocess_predict_data(test_data)
        if self.model is None:
            print("Model is not loaded. Please load the model first.")
            return
        predictions = self.model.predict(preprocess_data)
        labels = self.label_encoder.inverse_transform(predictions)
        return predictions, labels



if __name__ == "__main__":
    model = HotelPredictionModel()
    data = model.load_data('Dataset_B_hotel.csv')
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.save_model(filename='model_without_cloud.pkl',cloud = False)
    model.save_model(filename='model_with_cloud.pkl',cloud = True)

    # Prediction 
    new_data = pd.read_csv('Dataset_B_hotel.csv').dropna().head(1) # As Testing Data
    preds, labels = model.predict(new_data)
    print("Use Trained Model")
    print(preds)
    print(labels)
    # Use Saved Model (Cloud)
    print("Use Saved Model")
    model_with_cloud = HotelPredictionModel.load_model('model_with_cloud.pkl',cloud=True)
    new_data = pd.read_csv('Dataset_B_hotel.csv').dropna().head(1) # As Testing Data
    preds, labels = model_with_cloud.predict(new_data)
    print("With Cloud")
    print(preds)
    print(labels)

    # Use Saved Model (Without Cloud)
    model_without_cloud = HotelPredictionModel.load_model('model_without_cloud.pkl',cloud=False)
    new_data = pd.read_csv('Dataset_B_hotel.csv').dropna().head(1) # As Testing Data
    preds, labels = model_without_cloud.predict(new_data)
    print("Without Cloud")
    print(preds)
    print(labels)

