import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tkinter as tk
from PIL import Image, ImageTk

diabetes_dataset = pd.read_csv('diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

app = tk.Tk()
app.title("Diabetes Prediction")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((screen_width, screen_height)) 
background = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(app, image=background)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

def predict_diabetes():
    input_data = [pregnancies.get(), glucose.get(), blood_pressure.get(), skin_thickness.get(), insulin.get(),
                  bmi.get(), diabetes_pedigree_function.get(), age.get()]
    
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)
    
    if prediction[0] == 0:
        result_label.config(text="The person is not diabetic")
    else:
        result_label.config(text="The person is diabetic")

# GUI components
title_label = tk.Label(app, text="Diabetes Prediction", font=("Helvetica", 24))
title_label.place(relx=0.5, rely=0.1, anchor="center")      
pregnancies_label = tk.Label(app, text="Pregnancies:")
pregnancies_label.place(x=screen_width//4, y=screen_height//4)
pregnancies = tk.Entry(app)
pregnancies.place(x=screen_width//2, y=screen_height//4)

glucose_label = tk.Label(app, text="Glucose:")
glucose_label.place(x=screen_width//4, y=screen_height//4 + 30)
glucose = tk.Entry(app)
glucose.place(x=screen_width//2, y=screen_height//4 + 30)

blood_pressure_label = tk.Label(app, text="Blood Pressure:")
blood_pressure_label.place(x=screen_width//4, y=screen_height//4 + 60)
blood_pressure = tk.Entry(app)
blood_pressure.place(x=screen_width//2, y=screen_height//4 + 60)

skin_thickness_label = tk.Label(app, text="Skin Thickness:")
skin_thickness_label.place(x=screen_width//4, y=screen_height//4 + 90)
skin_thickness = tk.Entry(app)
skin_thickness.place(x=screen_width//2, y=screen_height//4 + 90)

insulin_label = tk.Label(app, text="Insulin:")
insulin_label.place(x=screen_width//4, y=screen_height//4 + 120)
insulin = tk.Entry(app)
insulin.place(x=screen_width//2, y=screen_height//4 + 120)

bmi_label = tk.Label(app, text="BMI:")
bmi_label.place(x=screen_width//4, y=screen_height//4 + 150)
bmi = tk.Entry(app)
bmi.place(x=screen_width//2, y=screen_height//4 + 150)

diabetes_pedigree_function_label = tk.Label(app, text="Diabetes Pedigree Function:")
diabetes_pedigree_function_label.place(x=screen_width//4, y=screen_height//4 + 180)
diabetes_pedigree_function = tk.Entry(app)
diabetes_pedigree_function.place(x=screen_width//2, y=screen_height//4 + 180)

age_label = tk.Label(app, text="Age:")
age_label.place(x=screen_width//4, y=screen_height//4 + 210)
age = tk.Entry(app)
age.place(x=screen_width//2, y=screen_height//4 + 210)

predict_button = tk.Button(app, text="Predict", command=predict_diabetes)
predict_button.place(x=screen_width//4, y=screen_height//4 + 250)

result_label = tk.Label(app, text="")
result_label.place(x=screen_width//4, y=screen_height//4 + 290)

app.geometry(f"{screen_width}x{screen_height}")  
app.mainloop()
