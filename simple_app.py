import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from faker import Faker
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bank_marketing.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
fake = Faker()

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

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    records_count = db.Column(db.Integer)
    predictions_count = db.Column(db.Integer)

def generate_realistic_data(num_records=1000):
    """Generate realistic bank marketing data for the past 10 years"""
    data = []
    
    # Job categories with realistic distributions
    jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    job_weights = [0.15, 0.12, 0.08, 0.05, 0.20, 0.10, 0.08, 0.12, 0.05, 0.10, 0.05]
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

def predict_customer_simple(customer_data):
    """Simple prediction based on customer data"""
    age = customer_data.get('age', 45)
    balance = customer_data.get('balance', 1000)
    duration = customer_data.get('duration', 300)
    campaign = customer_data.get('campaign', 1)

    score = 0
    if age > 50: score += 1
    if balance > 5000: score += 2
    if duration > 600: score += 1
    if campaign == 1: score += 1

    if score >= 3:
        prediction = 'yes'
        confidence = 0.8
    elif score >= 2:
        prediction = 'yes'
        confidence = 0.6
    elif score >= 1:
        prediction = 'no'
        confidence = 0.7
    else:
        prediction = 'no'
        confidence = 0.9

    return prediction, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get statistics
    total_customers = Customer.query.count()
    total_predictions = UploadedFile.query.with_entities(db.func.sum(UploadedFile.predictions_count)).scalar() or 0
    positive_predictions = Customer.query.filter_by(response='yes').count()
    negative_predictions = Customer.query.filter_by(response='no').count()
    
    # Get recent uploads
    recent_uploads = UploadedFile.query.order_by(UploadedFile.upload_date.desc()).limit(5).all()
    
    # Create real charts from database data
    charts = {}
    
    if total_customers > 0:
        # Age distribution chart
        age_data = db.session.query(Customer.age).all()
        ages = [row[0] for row in age_data]
        
        # Create age bins
        age_bins = [18, 25, 35, 45, 55, 65, 95]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        age_counts = []
        
        for i in range(len(age_bins) - 1):
            count = sum(1 for age in ages if age_bins[i] <= age < age_bins[i+1])
            age_counts.append(count)
        
        charts['age_distribution'] = json.dumps({
            'data': [{'x': age_labels, 'y': age_counts, 'type': 'bar', 'name': 'Age Distribution'}],
            'layout': {'title': 'Age Distribution of Customers', 'xaxis': {'title': 'Age Range'}, 'yaxis': {'title': 'Count'}}
        })
        
        # Job response chart
        job_response_data = db.session.query(Customer.job, Customer.response, db.func.count(Customer.id)).\
            group_by(Customer.job, Customer.response).all()
        
        job_data = {}
        for job, response, count in job_response_data:
            if job not in job_data:
                job_data[job] = {'yes': 0, 'no': 0}
            job_data[job][response] = count
        
        # Get top 8 jobs by total count
        job_totals = {job: sum(counts.values()) for job, counts in job_data.items()}
        top_jobs = sorted(job_totals.items(), key=lambda x: x[1], reverse=True)[:8]
        
        jobs = [job for job, _ in top_jobs]
        yes_counts = [job_data[job]['yes'] for job in jobs]
        no_counts = [job_data[job]['no'] for job in jobs]
        
        charts['job_response'] = json.dumps({
            'data': [
                {'x': jobs, 'y': yes_counts, 'type': 'bar', 'name': 'Yes'},
                {'x': jobs, 'y': no_counts, 'type': 'bar', 'name': 'No'}
            ],
            'layout': {'title': 'Response Rate by Job Category', 'barmode': 'group'}
        })
        
        # Balance distribution chart
        balances_yes = [c.balance for c in Customer.query.filter_by(response='yes').all()]
        balances_no = [c.balance for c in Customer.query.filter_by(response='no').all()]
        
        charts['balance_response'] = json.dumps({
            'data': [
                {'y': balances_yes, 'type': 'box', 'name': 'Yes'},
                {'y': balances_no, 'type': 'box', 'name': 'No'}
            ],
            'layout': {'title': 'Balance Distribution by Response'}
        })
    else:
        # Fallback to sample data if no records
        charts = {
            'age_distribution': json.dumps({
                'data': [{'x': [25, 35, 45, 55, 65], 'y': [20, 30, 25, 15, 10], 'type': 'bar', 'name': 'Age Distribution'}],
                'layout': {'title': 'Age Distribution of Customers', 'xaxis': {'title': 'Age'}, 'yaxis': {'title': 'Count'}}
            }),
            'job_response': json.dumps({
                'data': [
                    {'x': ['Management', 'Technician', 'Services', 'Admin'], 'y': [65, 45, 35, 55], 'type': 'bar', 'name': 'Yes'},
                    {'x': ['Management', 'Technician', 'Services', 'Admin'], 'y': [35, 55, 65, 45], 'type': 'bar', 'name': 'No'}
                ],
                'layout': {'title': 'Response Rate by Job Category', 'barmode': 'group'}
            }),
            'balance_response': json.dumps({
                'data': [
                    {'y': [2000, 3000, 4000, 5000], 'type': 'box', 'name': 'Yes'},
                    {'y': [1000, 1500, 2000, 2500], 'type': 'box', 'name': 'No'}
                ],
                'layout': {'title': 'Balance Distribution by Response'}
            })
        }
    
    return render_template('dashboard.html', 
                         total_customers=total_customers,
                         total_predictions=total_predictions,
                         positive_predictions=positive_predictions,
                         negative_predictions=negative_predictions,
                         recent_uploads=recent_uploads,
                         charts=charts)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
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
                    prediction, confidence = predict_customer_simple(customer_data)
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
    customers = Customer.query.all()
    
    data = []
    for customer in customers:
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
            'response': customer.response,
            'created_at': customer.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'historical_data.csv')
    df.to_csv(output_path, index=False)
    
    return send_file(output_path, as_attachment=True, download_name='historical_data.csv')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Bank Marketing Campaign Predictor...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 