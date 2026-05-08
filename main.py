from flask import Flask, render_template, request, redirect, session, url_for
import numpy as np
import pandas as pd
import pickle
import sqlite3
from flask_bcrypt import Bcrypt

# ================= FLASK APP SETUP =================
app = Flask(__name__)
app.secret_key = "secret123"
bcrypt = Bcrypt(app)

# ================= DATABASE CONNECTION =================
def get_db():
    conn = sqlite3.connect("users.db")
    return conn

# DATABASE TABLE & DEFAULT ADMIN FUNCTION
def init_db():
    db = get_db()
    
    db.execute('''CREATE TABLE IF NOT EXISTS users 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                   username TEXT UNIQUE, 
                   email TEXT UNIQUE, 
                   password TEXT)''')
    
    
    admin_check = db.execute("SELECT * FROM users WHERE username='admin'").fetchone()
    if not admin_check:
        hashed_pw = bcrypt.generate_password_hash("admin123").decode('utf-8')
        db.execute("INSERT INTO users (username, email, password) VALUES (?,?,?)", 
                   ('admin', 'admin@gmail.com', hashed_pw))
        db.commit()
    db.close()

#DB CHECK WHENT APP STARTING
init_db()

# ================= LOAD ML DATA & MODEL =================
try:
    precautions = pd.read_csv("Model/precautions_df.csv")
    description = pd.read_csv("Model/description.csv")
    medications = pd.read_csv("Model/medications.csv")
    svc = pickle.load(open('Model/svc.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {e}")

# ================= SYMPTOMS & DISEASES DICTIONARY =================
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14,
    'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17,
    'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
    'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31,
    'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34,
    'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43,
    'acute_liver_failure': 44, 'fluid_overload': 45,
    'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52,
    'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63,
    'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67,
    'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic_patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
    'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121,
    'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128,
    'blister': 129, 'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}

diseases_list = {
    15:'Fungal infection', 4:'Allergy', 16:'GERD', 9:'Chronic cholestasis',
    14:'Drug Reaction', 33:'Peptic ulcer diseae', 1:'AIDS', 12:'Diabetes',
    17:'Gastroenteritis', 6:'Bronchial Asthma', 23:'Hypertension',
    30:'Migraine', 7:'Cervical spondylosis', 32:'Paralysis (brain hemorrhage)',
    28:'Jaundice', 29:'Malaria', 8:'Chicken pox', 11:'Dengue',
    37:'Typhoid', 40:'Hepatitis A', 19:'Hepatitis B', 20:'Hepatitis C',
    21:'Hepatitis D', 22:'Hepatitis E', 3:'Alcoholic hepatitis',
    36:'Tuberculosis', 10:'Common Cold', 34:'Pneumonia',
    13:'Piles', 18:'Heart attack', 39:'Varicose veins',
    26:'Hypothyroidism', 24:'Hyperthyroidism', 25:'Hypoglycemia',
    31:'Osteoarthritis', 5:'Arthritis', 0:'Vertigo',
    2:'Acne', 38:'UTI', 35:'Psoriasis', 27:'Impetigo'
}

# ================= HELPER FUNCTIONS =================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [m for m in med.values]
    return desc, pre, med

def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1
    pred = svc.predict([input_vector])[0]
    return diseases_list[pred]

# ================= ROUTES =================

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")

@app.route("/contact")
def contact():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("contact.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if "user" in session and session["user"] == "admin":
        db = get_db()
        
        query = "SELECT id, username, email FROM users ORDER BY id ASC"
        users = db.execute(query).fetchall()
        db.close()
        return render_template("admin_dashboard.html", users=users)
    else:
        return "<h1 style='color:red; text-align:center; margin-top:50px;'>Access Denied!</h1>", 403

@app.route("/admin/delete/<int:id>")
def delete_user(id):
    if "user" in session and session["user"] == "admin":
        db = get_db()
        user_to_del = db.execute("SELECT username FROM users WHERE id=?", (id,)).fetchone()
        
        if user_to_del and user_to_del[0] != 'admin':
            db.execute("DELETE FROM users WHERE id=?", (id,))
            db.commit()
        db.close()
    return redirect(url_for("admin_dashboard"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip().lower()
        email = request.form["email"].strip()
        password = request.form["password"]
        
        if username == "admin":
            return "<script>alert('Error: admin name is reserved!'); window.history.back();</script>"
        
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, email, password) VALUES (?,?,?)", 
                       (username, email, hashed))
            db.commit()
            db.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return "<script>alert('Username or Email already exists!'); window.history.back();</script>"
            
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip().lower()
        password = request.form["password"]
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        db.close()
        
        if user:
            # BCRYPT PASSWORD CHEC
            if bcrypt.check_password_hash(user[3], password):
                session["user"] = username
                return redirect(url_for("index"))
            else:
                return "<script>alert('Invalid password'); window.history.back();</script>"
        else:
            return "<script>alert('User not found'); window.history.back();</script>"
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))
    symptoms = request.form.get("symptoms")
    if not symptoms:
        return redirect(url_for("index"))
        
    user_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",")]
    try:
        disease = get_predicted_value(user_symptoms)
        desc, pre, med = helper(disease)
        return render_template("index.html", predicted_disease=disease, dis_des=desc, my_precautions=pre[0] if pre else [], medications=med)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)