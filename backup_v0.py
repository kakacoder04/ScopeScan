from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from functools import wraps
from datetime import datetime
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import base64

# --------------------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'kakadziuhtrangquadi'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/endoscan'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --------------------------------------------------------------------------------------
model = YOLO("./model/model_detect.pt")

# --------------------------------------------------------------------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kakawiner04@gmail.com'
app.config['MAIL_PASSWORD'] = 'qtti isvz tqoa lftb'
mail = Mail(app)

# --------------------------------------------------------------------------------------
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    admin = db.Column(db.Boolean, default=False)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    condition = db.Column(db.String(100), nullable=False)
    date_added = db.Column(db.Date, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)

# --------------------------------------------------------------------------------------
with app.app_context():
    db.create_all()

    if not Users.query.filter_by(email="kakawiner04@gmail.com").first():
        new_user = Users(name="Phan Ka Ka", email="kakawiner04@gmail.com", password="1", admin=True)
        db.session.add(new_user)
        db.session.commit()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to continue.', 'warning')
            return redirect(url_for('login'))
        if not session.get('admin', False):
            flash('Access denied. Administrators only.', 'warning')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --------------------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        if session.get('admin', False):
            return redirect(url_for('admin'))
        else:
            return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = Users.query.filter_by(email=email).first()
        if user and user.password == password:
            session['email'] = email
            session['admin'] = user.admin
            session['user_id'] = user.id
            if user.admin:
                return redirect(url_for('admin'))
            return redirect(url_for('dashboard'))
        
        session.pop('_flashes', None)
        flash('Invalid email or password!', 'warning')
    
    return render_template('index.html')

# --------------------------------------------------------------------------------------
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    user = Users.query.get(session['user_id'])

    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        name = request.form.get('name')
        date_added = request.form.get('date_added')
        condition = request.form.get('condition')
        image_file = request.files.get('image_file')

        if not image_file or image_file.filename == '':
            flash('Please select an image file.', 'warning')
            return redirect(url_for('dashboard'))

        if image_file and allowed_file(image_file.filename):
            upload_dir = os.path.join(app.static_folder, 'images')
            os.makedirs(upload_dir, exist_ok=True)
            
            base_name = f"{condition.replace(' ', '')}_{name.replace(' ', '')}"
            extension = os.path.splitext(image_file.filename)[1]
            filename = get_unique_filename(base_name, extension, upload_dir)
            image_path = os.path.join(upload_dir, filename)
            image_file.save(image_path)
            image_url = f"images/{filename}"

            new_patient = Patient(
                patient_id=patient_id,
                name=name,
                date_added=datetime.strptime(date_added, '%Y-%m-%d').date(),
                condition=condition,
                image_path=image_url
            )
            db.session.add(new_patient)
            db.session.commit()
            
            flash('Patient added successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid file type. Please upload an image.', 'warning')
            return redirect(url_for('dashboard'))

    return render_template('dashboard.html', user=user)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def get_unique_filename(base_name, extension, upload_dir):
    filename = secure_filename(base_name + extension)
    filepath = os.path.join(upload_dir, filename)
    counter = 1
    while os.path.exists(filepath):
        new_filename = secure_filename(f"{base_name}_{counter}{extension}")
        filepath = os.path.join(upload_dir, new_filename)
        filename = new_filename
        counter += 1
    return filename

# --------------------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
@login_required
def analyze_image():
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        image_file.seek(0)
        img_bytes = io.BytesIO(image_file.read())
        img = Image.open(img_bytes)
        
        if img.size[0] > 640:  
            img = img.resize((640, int(640 * img.size[1] / img.size[0])))
        
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            pass
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        results = model.predict(img, verbose=False, conf=0.25, imgsz=640) 
        
        annotated_img = img_array.copy()
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            op = "Normal"
            confidence = 100.0
            boxes_list = []
        else:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            class_names = ["Normal", "Polyp", "Esophagitis", "Ulcerative Colitis"]
            
            max_conf_idx = np.argmax(confidences)
            op = class_names[int(classes[max_conf_idx])] if int(classes[max_conf_idx]) < len(class_names) else "Unknown"
            confidence = confidences[max_conf_idx] * 100
            
            boxes_list = []
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                class_id = int(classes[i])
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                label = f"{class_name}: {confidences[i]*100:.1f}%"
                cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                boxes_list.append({
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3]),
                    'confidence': float(confidences[i] * 100),
                    'class': class_name
                })
        
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        descriptions = {
            "Normal": "The image shows normal esophageal tissue with no signs of abnormality.",
            "Ulcerative Colitis": "Ulcerative colitis is an inflammatory bowel disease that causes long-lasting inflammation and ulcers in the digestive tract.",
            "Polyp": "A polyp is an abnormal growth of tissue projecting from a mucous membrane. In the colon, polyps can be precancerous.",
            "Esophagitis": "Esophagitis is inflammation of the esophagus, often caused by acid reflux or infection.",
            "Unknown": "Detection result is unclear. Please consult a specialist."
        }
        
        treatments = {
            "Normal": "No treatment required. Regular check-ups recommended.",
            "Ulcerative Colitis": "Anti-inflammatory drugs, immunosuppressants, or biologics. Surgery in severe cases.",
            "Polyp": "Polypectomy during colonoscopy. Follow-up screenings to monitor for recurrence.",
            "Esophagitis": "Antacids, proton pump inhibitors, or dietary changes to reduce acid reflux.",
            "Unknown": "Further medical evaluation required."
        }
        
        return jsonify({
            'condition': op,
            'confidence': f"{confidence:.2f}%",
            'description': descriptions.get(op, "No description available."),
            'treatment': treatments.get(op, "No treatment recommendation available."),
            'boxes': boxes_list,
            'annotated_image': f"data:image/jpeg;base64,{annotated_base64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------------------------------------------
@app.route('/analyze_video', methods=['POST'])
@login_required
def analyze_video():
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image_file']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        image_file.seek(0)
        img_bytes = io.BytesIO(image_file.read())
        img = Image.open(img_bytes)
        
        if img.size[0] > 320:  
            img = img.resize((320, int(320 * img.size[1] / img.size[0])))
        
        img_array = np.array(img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            pass
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        results = model.predict(img, verbose=False, conf=0.3, imgsz=320)  
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            op = "Normal"
            confidence = 100.0
            boxes_list = []
        else:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            class_names = ["Normal", "Polyp", "Esophagitis", "Ulcerative Colitis"]
            
            max_conf_idx = np.argmax(confidences)
            op = class_names[int(classes[max_conf_idx])] if int(classes[max_conf_idx]) < len(class_names) else "Unknown"
            confidence = confidences[max_conf_idx] * 100
            
            boxes_list = []
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().astype(int)
                class_id = int(classes[i])
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                
                boxes_list.append({
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3]),
                    'confidence': float(confidences[i] * 100),
                    'class': class_name
                })
        
        return jsonify({
            'condition': op,
            'confidence': f"{confidence:.2f}%",
            'boxes': boxes_list
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------------------------------------------------
@app.route('/video', methods=['GET', 'POST'])
@login_required
def video():
    user = Users.query.get(session['user_id'])
    return render_template('video.html', user=user)

# --------------------------------------------------------------------------------------
@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    users = Users.query.all()
    return render_template('admin.html', users=users)

# --------------------------------------------------------------------------------------
@app.route('/newaccount', methods=['GET', 'POST'])
@admin_required
def newaccount():
    session.pop('_flashes', None)

    if request.method == 'POST':
        name = request.form.get('fullName')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        send_welcome = 'send_welcome' in request.form
        
        existing_user = Users.query.filter_by(email=email).first()
        if existing_user:
            flash('Email address already exists.', 'warning')
            return render_template('newaccount.html')
        
        admin_status = role == 'admin'
        
        new_user = Users(name=name, email=email, password=password, admin=admin_status)
        db.session.add(new_user)
        db.session.commit()
        
        if send_welcome:
            msg = Message(
                subject='Welcome to ScopeScan!',
                sender=app.config['MAIL_USERNAME'],
                recipients=[email],
                body=f'Dear {name},\n\nYour account has been created successfully!\n\nEmail: {email}\nPassword: {password}\nRole: {role}\n\nPlease use these credentials to log in to ScopeScan.\n\nBest regards,\nScopeScan Team',
                html=f'''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Welcome to ScopeScan!</title>
                </head>
                <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f4f4f4; color: #333333;">
                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f4f4f4;">
                        <tr>
                            <td style="padding: 20px 0;">
                                <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" width="600" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                                    <!-- Header -->
                                    <tr>
                                        <td style="background-color: #007bff; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
                                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: bold;">Welcome to ScopeScan!</h1>
                                        </td>
                                    </tr>
                                    <!-- Content -->
                                    <tr>
                                        <td style="padding: 40px 30px;">
                                            <p style="margin: 0 0 20px; font-size: 16px; line-height: 1.5;">Dear <strong>{name}</strong>,</p>
                                            <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.5;">Your account has been created successfully!</p>
                                            
                                            <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="width: 100%; margin-bottom: 30px;">
                                                <tr>
                                                    <td style="padding: 10px 0; border-bottom: 1px solid #e0e0e0;">
                                                        <strong style="color: #007bff;">Email:</strong> {email}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td style="padding: 10px 0; border-bottom: 1px solid #e0e0e0;">
                                                        <strong style="color: #007bff;">Password:</strong> {password}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <td style="padding: 10px 0;">
                                                        <strong style="color: #007bff;">Role:</strong> {role}
                                                    </td>
                                                </tr>
                                            </table>
                                            
                                            <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.5;">Please use these credentials to log in to ScopeScan.</p>
                                            
                                        </td>
                                    </tr>
                                    <!-- Footer -->
                                    <tr>
                                        <td style="background-color: #f8f9fa; padding: 20px 30px; text-align: center; border-radius: 0 0 8px 8px; font-size: 14px; color: #6c757d;">
                                            <p style="margin: 0 0 10px;">Best regards,<br><strong>ScopeScan Team</strong></p>
                                            <p style="margin: 0;">&copy; 2025 ScopeScan. All rights reserved.</p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                    </table>
                </body>
                </html>
                '''
            )
            try:
                mail.send(msg)
            except Exception as e:
                flash(f'Account created but failed to send welcome email: {str(e)}', 'warning')
                return redirect(url_for('admin'))
        
        flash('Account created successfully!', 'success')
        return redirect(url_for('admin'))
    
    return render_template('newaccount.html')

# --------------------------------------------------------------------------------------
@app.route('/update/<int:user_id>', methods=['POST'])
@admin_required
def update_user(user_id):
    user = Users.query.get_or_404(user_id)
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role')
    
    if not all([name, email, role]):
        flash('Please fill in all required fields.', 'warning')
        return redirect(url_for('admin'))
    
    existing_user = Users.query.filter(Users.email == email, Users.id != user_id).first()
    if existing_user:
        flash('Email address already exists.', 'warning')
        return redirect(url_for('admin'))
    
    user.name = name
    user.email = email
    if password:
        user.password = password
    else:
        password = "Nothing changes!"
    
    user.admin = (role == 'admin')

    msg = Message(
        subject='Welcome to ScopeScan!',
        sender=app.config['MAIL_USERNAME'],
        recipients=[email],
        body=f'Dear {name},\n\nYour account has been changed successfully!\n\nEmail: {email}\nPassword: {password}\nRole: {role}\n\nPlease use these credentials to log in to ScopeScan.\n\nBest regards,\nScopeScan Team',
        html=f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to ScopeScan!</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f4f4f4; color: #333333;">
            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f4f4f4;">
                <tr>
                    <td style="padding: 20px 0;">
                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center" width="600" style="background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                            <!-- Header -->
                            <tr>
                                <td style="background-color: #007bff; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
                                    <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: bold;">Welcome to ScopeScan!</h1>
                                </td>
                            </tr>
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px 30px;">
                                    <p style="margin: 0 0 20px; font-size: 16px; line-height: 1.5;">Dear <strong>{name}</strong>,</p>
                                    <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.5;">Your account has been changed successfully!</p>
                                    
                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" style="width: 100%; margin-bottom: 30px;">
                                        <tr>
                                            <td style="padding: 10px 0; border-bottom: 1px solid #e0e0e0;">
                                                <strong style="color: #007bff;">Email:</strong> {email}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 10px 0; border-bottom: 1px solid #e0e0e0;">
                                                <strong style="color: #007bff;">Password:</strong> {password}
                                            </td>
                                        </tr>
                                        <tr>
                                            <td style="padding: 10px 0;">
                                                <strong style="color: #007bff;">Role:</strong> {role}
                                            </td>
                                        </tr>
                                    </table>
                                    
                                    <p style="margin: 0 0 30px; font-size: 16px; line-height: 1.5;">Please use these credentials to log in to ScopeScan.</p>
                                    
                                </td>
                            </tr>
                            <!-- Footer -->
                            <tr>
                                <td style="background-color: #f8f9fa; padding: 20px 30px; text-align: center; border-radius: 0 0 8px 8px; font-size: 14px; color: #6c757d;">
                                    <p style="margin: 0 0 10px;">Best regards,<br><strong>ScopeScan Team</strong></p>
                                    <p style="margin: 0;">&copy; 2025 ScopeScan. All rights reserved.</p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        '''
    )
    try:
        mail.send(msg)
    except Exception as e:
        flash(f'Failed to send changed email: {str(e)}', 'warning')
        return redirect(url_for('admin'))
    
    db.session.commit()
    flash('Account updated successfully!', 'success')
    return redirect(url_for('admin'))

@app.route('/delete/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = Users.query.get_or_404(user_id)
    if user.id == session['user_id']:
        flash('Cannot delete your own account.', 'warning')
        return redirect(url_for('admin'))
    db.session.delete(user)
    db.session.commit()
    flash('Account deleted successfully!', 'success')
    return redirect(url_for('admin'))

# --------------------------------------------------------------------------------------
@app.route('/endoscopy', methods=['GET'])
@admin_required
def endoscopy():
    patients = Patient.query.order_by(Patient.date_added.desc()).all()
    return render_template('endoscopy.html', patients=patients)

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
@admin_required
def delete_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.image_path:
        full_path = os.path.join(app.static_folder, patient.image_path)
        if os.path.exists(full_path):
            os.remove(full_path)
    db.session.delete(patient)
    db.session.commit()
    flash('Patient and image deleted successfully!', 'success')
    return redirect(url_for('endoscopy'))

# --------------------------------------------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    from pyngrok import ngrok

    public_url = ngrok.connect(6000)
    print("Ngrok public URL:", public_url)

    app.run(host="0.0.0.0", port=6000, debug=False)