import pandas as pd
import joblib
 
# Load the saved model
model_path = 'sgd_model.joblib'
clf = joblib.load(model_path)
 
# Function to prepare a single file for prediction
def prepare_single_file(file_path):
    data = pd.read_csv(file_path)
    
    # Drop unnecessary columns if they exist
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
 
    # Extract the ShapeID from the first row
    shape_id = data['ShapeID'].iloc[0]
    print(f"Processing shape ID: {shape_id} from file: {file_path}")
    
    # Feature Engineering: Calculate mean and std radius
    data['radius'] = ((data['X']**2 + data['Y']**2)**0.5)
    features = data.groupby('ShapeID').agg({
        'radius': ['mean', 'std']
    }).reset_index()
    features.columns = ['ShapeID', 'mean_radius', 'std_radius']
    
    return features[['mean_radius', 'std_radius']]
 
# Example of making a prediction on a single file
file_path = 'isolated.csv'
X_new = prepare_single_file(file_path)
prediction = clf.predict(X_new)
 
# Output the prediction
print(f'Prediction for shape ID {X_new.index[0]}: {"Circle" if prediction[0] == 1 else "Not Circle"}')