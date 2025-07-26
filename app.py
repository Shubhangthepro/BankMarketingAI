import os
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bank_marketing.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
fake = Faker()

# Database Models
class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    job = db.Column(db.String(50))
    marital = db.Column(db.String(20))
    education = db.Column(db.String(30))
    balance = db.Column(db.Float)
    housing = db.Column(db.String(10))
    loan = db.Column(db.String(10))
    contact = db.Column(db.String(20))
    month = db.Column(db.String(10))
    day_of_week = db.Column(db.String(10))
    duration = db.Column(db.Integer)
    campaign = db.Column(db.Integer)
    poutcome = db.Column(db.String(20))
    response = db.Column(db.String(5))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'))
    predicted_response = db.Column(db.String(5))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    customer = db.relationship('Customer', backref='predictions')

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    records_count = db.Column(db.Integer)
    predictions_count = db.Column(db.Integer)

# Global variables for model and encoders
model = None
label_encoders = {}
scaler = None

def generate_realistic_data(num_records=10000):
    """Generate realistic bank marketing data for the past 10 years"""
    data = []
    
    # Job categories with realistic distributions
    jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    job_weights = [0.15, 0.12, 0.08, 0.05, 0.20, 0.10, 0.08, 0.12, 0.05, 0.10, 0.05]
    # Normalize weights to sum to 1
    job_weights = np.array(job_weights) / sum(job_weights)
    
    # Marital status
    marital_status = ['married', 'single', 'divorced']
    marital_weights = [0.55, 0.35, 0.10]
    marital_weights = np.array(marital_weights) / sum(marital_weights)
    
    # Education levels
    education_levels = ['primary', 'secondary', 'tertiary', 'unknown']
    education_weights = [0.15, 0.45, 0.35, 0.05]
    education_weights = np.array(education_weights) / sum(education_weights)
    
    # Contact types
    contact_types = ['cellular', 'telephone', 'unknown']
    contact_weights = [0.70, 0.25, 0.05]
    contact_weights = np.array(contact_weights) / sum(contact_weights)
    
    # Months
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Days of week
    days = ['mon', 'tue', 'wed', 'thu', 'fri']
    
    # Previous outcome
    poutcomes = ['failure', 'success', 'other', 'unknown']
    poutcome_weights = [0.60, 0.15, 0.10, 0.15]
    poutcome_weights = np.array(poutcome_weights) / sum(poutcome_weights)
    
    # Calculate date range for past 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)  # 10 years ago
    
    for i in range(num_records):
        # Generate realistic age distribution
        age = int(np.random.normal(45, 15))
        age = max(18, min(95, age))
        
        # Generate realistic balance distribution (skewed towards lower values)
        balance = np.random.exponential(1000)
        balance = min(balance, 100000)  # Cap at 100k
        
        # Generate realistic duration (call duration in seconds)
        duration = int(np.random.exponential(300))
        duration = max(60, min(3600, duration))  # Between 1 minute and 1 hour
        
        # Generate campaign number (number of contacts)
        campaign = np.random.poisson(3) + 1  # Most people contacted 1-5 times
        
        # Generate response based on realistic factors
        response_prob = 0.1  # Base probability
        
        # Increase probability based on positive factors
        if age > 50: response_prob += 0.05
        if balance > 5000: response_prob += 0.08
        if duration > 600: response_prob += 0.03
        if campaign == 1: response_prob += 0.02
        
        response = 'yes' if np.random.random() < response_prob else 'no'
        
        # Generate realistic date distributed over past 10 years
        # More recent dates should have higher probability (exponential decay)
        days_ago = np.random.exponential(365*5)  # Mean of 5 years ago
        days_ago = min(days_ago, 365*10)  # Cap at 10 years
        created_at = end_date - timedelta(days=int(days_ago))
        
        # Adjust month and day based on created_at date
        month = created_at.strftime('%b').lower()[:3]
        day_of_week = created_at.strftime('%a').lower()[:3]
        
        record = {
            'age': age,
            'job': np.random.choice(jobs, p=job_weights),
            'marital': np.random.choice(marital_status, p=marital_weights),
            'education': np.random.choice(education_levels, p=education_weights),
            'balance': round(balance, 2),
            'housing': np.random.choice(['yes', 'no'], p=[0.75, 0.25]),
            'loan': np.random.choice(['yes', 'no'], p=[0.15, 0.85]),
            'contact': np.random.choice(contact_types, p=contact_weights),
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'poutcome': np.random.choice(poutcomes, p=poutcome_weights),
            'response': response,
            'created_at': created_at
        }
        data.append(record)
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for training"""
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle categorical variables
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    numerical_columns = ['age', 'balance', 'duration', 'campaign']
    scaler = StandardScaler()
    df_processed[numerical_columns] = scaler.fit_transform(df_processed[numerical_columns])
    
    return df_processed, scaler

def train_model():
    """Train the predictive model"""
    global model, label_encoders, scaler
    
    # Generate training data
    print("Generating training data...")
    df = generate_realistic_data(50000)  # 50k records for training
    
    # Preprocess data
    df_processed, scaler = preprocess_data(df)
    
    # Prepare features and target
    feature_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 
                      'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome']
    X = df_processed[feature_columns]
    y = df_processed['response']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models and select the best one
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model_candidate in models.items():
        model_candidate.fit(X_train, y_train)
        y_pred = model_candidate.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        if score > best_score:
            best_score = score
            best_model = model_candidate
            print(f"New best model: {name} with accuracy: {score:.4f}")
    
    model = best_model
    
    # Save the model and encoders
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"Model trained and saved with accuracy: {best_score:.4f}")
    return best_score

def load_model():
    """Load the trained model and encoders"""
    global model, label_encoders, scaler
    
    try:
        model = joblib.load('model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model loaded successfully")
        return True
    except FileNotFoundError:
        print("Model not found. Training new model...")
        train_model()
        return True

def predict_customer(customer_data):
    """Predict response for a single customer"""
    if model is None:
        load_model()
    
    # Preprocess the customer data
    df = pd.DataFrame([customer_data])
    
    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))
    
    # Apply scaling
    numerical_columns = ['age', 'balance', 'duration', 'campaign']
    df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    # Make prediction
    feature_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 
                      'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome']
    X = df[feature_columns]
    
    prediction = model.predict(X)[0]
    
    # Get probability and ensure it's valid
    try:
        proba = model.predict_proba(X)[0]
        confidence = max(proba) if len(proba) > 0 else 0.5
    except:
        confidence = 0.5
    
    return prediction, confidence

@app.route('/')
def home():
    """Homepage with animated UI"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with statistics and visualizations"""
    # Get statistics
    total_customers = Customer.query.count()
    total_predictions = Prediction.query.count()
    positive_predictions = Prediction.query.filter_by(predicted_response='yes').count()
    negative_predictions = Prediction.query.filter_by(predicted_response='no').count()
    
    # Get recent uploads
    recent_uploads = UploadedFile.query.order_by(UploadedFile.upload_date.desc()).limit(5).all()
    
    # Create visualizations
    charts = create_dashboard_charts()
    
    return render_template('dashboard.html', 
                         total_customers=total_customers,
                         total_predictions=total_predictions,
                         positive_predictions=positive_predictions,
                         negative_predictions=negative_predictions,
                         recent_uploads=recent_uploads,
                         charts=charts)

def create_dashboard_charts():
    """Create Plotly charts for dashboard"""
    charts = {}
    
    # Age distribution chart
    customers = Customer.query.all()
    if customers:
        ages = [c.age for c in customers]
        
        fig_age = px.histogram(x=ages, nbins=20, title="Age Distribution of Customers",
                              labels={'x': 'Age', 'y': 'Count'})
        fig_age.update_layout(showlegend=False)
        charts['age_distribution'] = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Response rate by job
        job_response_data = db.session.query(Customer.job, Customer.response, db.func.count(Customer.id)).\
            group_by(Customer.job, Customer.response).all()
        
        job_data = {}
        for job, response, count in job_response_data:
            if job not in job_data:
                job_data[job] = {'yes': 0, 'no': 0}
            job_data[job][response] = count
        
        jobs = list(job_data.keys())
        yes_counts = [job_data[job]['yes'] for job in jobs]
        no_counts = [job_data[job]['no'] for job in jobs]
        
        fig_job = go.Figure(data=[
            go.Bar(name='Yes', x=jobs, y=yes_counts),
            go.Bar(name='No', x=jobs, y=no_counts)
        ])
        fig_job.update_layout(title="Response Rate by Job Category", barmode='group')
        charts['job_response'] = json.dumps(fig_job, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Balance vs Response
        balances_yes = [c.balance for c in customers if c.response == 'yes']
        balances_no = [c.balance for c in customers if c.response == 'no']
        
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Box(y=balances_yes, name='Yes', boxpoints='outliers'))
        fig_balance.add_trace(go.Box(y=balances_no, name='No', boxpoints='outliers'))
        fig_balance.update_layout(title="Balance Distribution by Response")
        charts['balance_response'] = json.dumps(fig_balance, cls=plotly.utils.PlotlyJSONEncoder)
    
    return charts

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read and process the uploaded file
                df = pd.read_csv(filepath)
                required_columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 
                                  'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome']
                
                # Check if all required columns are present
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    flash(f'Missing required columns: {", ".join(missing_columns)}', 'error')
                    return redirect(request.url)
                
                # Clean the data by stripping whitespace from string columns
                string_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                
                # Make predictions
                predictions = []
                for _, row in df.iterrows():
                    customer_data = row.to_dict()
                    prediction, confidence = predict_customer(customer_data)
                    predictions.append({
                        'customer_data': customer_data,
                        'prediction': prediction,
                        'confidence': confidence
                    })
                
                # Store results in session for display
                session_data = {
                    'filename': filename,
                    'total_records': len(df),
                    'predictions': predictions
                }
                
                # Save to database
                uploaded_file = UploadedFile(
                    filename=filename,
                    records_count=len(df),
                    predictions_count=len(predictions)
                )
                db.session.add(uploaded_file)
                db.session.commit()
                
                return render_template('results.html', session_data=session_data)
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/generate_data')
def generate_data():
    """Generate and populate database with 10 years of realistic data"""
    try:
        # Clear existing customer data to avoid duplicates
        Customer.query.delete()
        db.session.commit()
        
        # Generate 100,000 records representing 10 years of data
        print("Generating 10 years of realistic bank marketing data...")
        df = generate_realistic_data(100000)
        
        # Add to database in batches for better performance
        batch_size = 1000
        total_records = len(df)
        
        for i in range(0, total_records, batch_size):
            batch = df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                customer = Customer(**row.to_dict())
                db.session.add(customer)
            
            db.session.commit()
            print(f"Processed {min(i+batch_size, total_records)}/{total_records} records...")
        
        flash(f'Successfully generated {len(df)} customer records spanning 10 years of historical data', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Error generating data: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/download_results')
def download_results():
    """Download prediction results as CSV"""
    predictions = Prediction.query.all()
    
    data = []
    for pred in predictions:
        customer = pred.customer
        data.append({
            'age': customer.age,
            'job': customer.job,
            'marital': customer.marital,
            'education': customer.education,
            'balance': customer.balance,
            'housing': customer.housing,
            'loan': customer.loan,
            'contact': customer.contact,
            'month': customer.month,
            'day_of_week': customer.day_of_week,
            'duration': customer.duration,
            'campaign': customer.campaign,
            'poutcome': customer.poutcome,
            'predicted_response': pred.predicted_response,
            'confidence': pred.confidence,
            'prediction_date': pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    df.to_csv(output_path, index=False)
    
    return send_file(output_path, as_attachment=True, download_name='predictions.csv')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.get_json()
        prediction, confidence = predict_customer(data)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000) 