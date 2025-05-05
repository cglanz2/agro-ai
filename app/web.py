# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
from flask import Flask, g, render_template, flash, redirect, url_for, session, request, jsonify
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from app import app, db
from app.models import User, Label, Image, Session
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sqlalchemy import desc, func
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from datetime import datetime
import pandas as pd
import os
import numpy as np
import boto3
from io import StringIO

bootstrap = Bootstrap(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # Redirect to login page if user is not logged in

def getData():
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    s3 = boto3.client('s3')
    path = 's3://agro-ai-maize/csvOut.csv'

    try:
        data = pd.read_csv(path, index_col = 0, header = None)
    except FileNotFoundError:
        print('Error: ' + path + ' not found.')
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']

    data_mod = data.astype({'8': 'int32','9': 'int32','10': 'int32','12': 'int32','14': 'int32'})
    return data_mod.iloc[:, :-1]

def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image

    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model, train_img_names

def renderLabel(form):
    """
    Prepares a render_template to show the label.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('label'), form = form, picture = img, confidence = session['confidence'])

def initializeAL(form, confidence_break = .7):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    confidence_break : number
        How confident the model is.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    preprocess = DataPreprocessing(True)
    ml_classifier = RandomForestClassifier()
    data = getData()
    al_model = Active_ML_Model(data, ml_classifier, preprocess)

    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)

    return renderLabel(form)

def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    test_set = data[data.index.isin(train_img_names) == False]

    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set, 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)

def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))
    
    # Prepare training data
    if session['train'] is not None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']
    
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    
    # Get confidence before inserting labels
    session['confidence'] = float(np.mean(ml_model.K_fold()))
    confidence_score = session['confidence']
    
    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic))
    else:
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        
        # Ensure user is logged in and user_id is available
        if not current_user.is_authenticated:
            return redirect(url_for('login'))

        # Retrieve user_id from the logged-in user (current_user)
        user_id = current_user.id
        
        new_session = Session(user_id=user_id)
        db.session.add(new_session)
        db.session.commit()
        
        user_confidence = session['confidence']
        
        save_to_database(health_pic_user, 'H', user_id, user_confidence, False, new_session.id)
        save_to_database(blight_pic_user, 'B', user_id, user_confidence, False, new_session.id)
        save_to_database(health_pic, 'H', user_id, health_pic_prob, True, new_session.id)
        save_to_database(blight_pic, 'B', user_id, blight_pic_prob, True, new_session.id)
        
        session.pop('model', None)
        
        return render_template('final.html', form = form, confidence = "{:.2%}".format(round(user_confidence,4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)

def save_to_database(image, label, user_id, confidence_score, model, session_id):
    for i, img in enumerate(image):
        # Check if the image already exists in the database
        existing_image = Image.query.filter_by(filename=img).first()
        if not existing_image:
            new_image = Image(filename=img, user_id=user_id, label=label)
            db.session.add(new_image)
            db.session.commit()
            image_id = new_image.id
        else:
            image_id = existing_image.id

        confidence = confidence_score[i] if model else confidence_score
        new_label = Label(text=label, image_id=image_id, user_id=user_id, confidence=confidence, model=model, session_id=session_id)
        db.session.add(new_label)
        db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Look for the user in the database
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            # Login the user and redirect to a protected page
            login_user(user)
            return redirect(url_for('label'))
        else:
            flash('Login failed. Please check your email and password and try again.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def calculate_accuracy(type, model, return_counts=False):
    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, 'data', 'csvOut.xlsx')
    df = pd.read_excel(file_path, usecols=[0, 16], header=None)
    df.columns = ['filename', 'true_label']
    ground_truth = dict(zip(df['filename'], df['true_label']))

    if not current_user.is_authenticated:
        flash('User not authenticated. Please log in.', 'danger')
        return redirect(url_for('login'))
    
    if type == "all":
        images = (db.session.query(Image.filename, Label.text, Session.id)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.model == model).all())
    elif type == "user":
        images = (db.session.query(Image.filename, Label.text, Session.id)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.user_id == current_user.id, Label.model == model).all())
        
    total = 0
    correct = 0
    incorrect_images = []

    for filename, label, session_id in images:
        true_label = ground_truth.get(filename)
        if true_label:
            total += 1
            if label.upper() == str(true_label).upper():
                correct += 1
            else:
                incorrect_images.append((filename, label, true_label, session_id))

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0
    return (accuracy, total, correct, incorrect_images) if return_counts else accuracy

def get_most_mislabeled_image():
    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, 'data', 'csvOut.xlsx')
    df = pd.read_excel(file_path, usecols=[0, 16], header=None)
    df.columns = ['filename', 'true_label']
    ground_truth = dict(zip(df['filename'], df['true_label']))

    all_images = Image.query.all()
    mislabel_counts = defaultdict(int)

    for img in all_images:
        true_label = ground_truth.get(img.filename)
        if true_label and img.label.upper() != str(true_label).upper():
            mislabel_counts[img.filename] += 1

    if not mislabel_counts:
        return None

    most_mislabeled = max(mislabel_counts.items(), key=lambda x: x[1])
    return {"filename": most_mislabeled[0], "count": most_mislabeled[1]}

@app.before_request
def before_request():
    if current_user.is_authenticated:
        latest_label = Label.query.filter_by(user_id=current_user.id).order_by(desc(Label.id)).first()
        g.latest_confidence = latest_label.confidence if latest_label else 0.0

def ordinal(n):
    return str(n) + (
        "th" if 11 <= n % 100 <= 13 else
        {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )

def format_timestamp(dt):
    return dt.strftime('%B {S}, %Y at %I:%M %p UTC').replace('{S}', ordinal(dt.day))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password']
        retype_password = request.form['retype-password']

        # Validate the passwords match
        if password != retype_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        # Check if the email already exists in the database
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists. Please log in.', 'danger')
            return redirect(url_for('login'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create the new user
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))  # Redirect to login page after successful registration

    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        email = request.form['email']
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        # Get the user from the database
        user = get_user_by_email(email)
        
        # Check if the user exists and the current password is correct
        if user and user.check_password(current_password):
            # Check if the new passwords match
            if new_password == confirm_password:
                # Update the user's password
                user.set_password(new_password)  # Make sure you have a method to hash and set the password
                
                # Commit the changes to the database
                db.session.commit()
                
                flash('Password updated successfully!', 'success')
            else:
                flash('New passwords do not match.', 'error')
        else:
            flash('Current password is incorrect.', 'error')

        return redirect(url_for('profile'))

    return render_template('profile.html')

@app.route('/saved')
@login_required
def saved():
    user_healthy_plants = (db.session.query(Image, Session.id, Session.timestamp)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.user_id == current_user.id, Label.text == 'H', Label.model == False).all())
    user_unhealthy_plants = (db.session.query(Image, Session.id, Session.timestamp)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.user_id == current_user.id, Label.text == 'B', Label.model == False).all())
    healthy_plants = (db.session.query(Image, Session.id, Session.timestamp)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.user_id == current_user.id, Label.text == 'H', Label.model == True).all())
    unhealthy_plants = (db.session.query(Image, Session.id, Session.timestamp)
                        .join(Label, Image.id == Label.image_id)      
                        .join(Session, Label.session_id == Session.id) 
                        .filter(Label.user_id == current_user.id, Label.text == 'B', Label.model == True).all())

    user_healthy_count = len(user_healthy_plants)
    user_unhealthy_count = len(user_unhealthy_plants)
    healthy_count = len(healthy_plants)
    unhealthy_count = len(unhealthy_plants)

    user_healthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}",
        'timestamp': format_timestamp(timestamp),
        'session_id': session_id
    } for plant, session_id, timestamp in user_healthy_plants]

    user_unhealthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}",
        'timestamp': format_timestamp(timestamp),
        'session_id': session_id
    } for plant, session_id, timestamp in user_unhealthy_plants]
    
    healthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}",
        'timestamp': format_timestamp(timestamp),
        'session_id': session_id
    } for plant, session_id, timestamp in healthy_plants]
    
    unhealthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}",
        'timestamp': format_timestamp(timestamp),
        'session_id': session_id
    } for plant, session_id, timestamp in unhealthy_plants]

    return render_template('saved.html', 
                           user_healthy_count=user_healthy_count,
                           user_unhealthy_count=user_unhealthy_count,
                           user_healthy_plants=user_healthy_plants_data,
                           user_unhealthy_plants=user_unhealthy_plants_data,
                           healthy_count=healthy_count,
                           unhealthy_count=unhealthy_count,
                           healthy_plants=healthy_plants_data,
                           unhealthy_plants=unhealthy_plants_data)

@app.route("/", methods=['GET'])
@app.route("/index.html", methods=['GET'])
def home():
    """
    Redirects users based on login status:
    - If logged in, redirect to label.html
    - If not logged in, redirect to index.html
    """
    if session.get('user_logged_in'):
        return render_template('label.html')
    else:
        return render_template('index.html')

@app.route("/label.html", methods=['GET', 'POST'])
@login_required
def label():
    """
    Operates the label(label.html) web page.
    """
    form = LabelForm()

    latest_label = Label.query.filter_by(user_id=current_user.id).order_by(desc(Label.id)).first()
    latest_confidence = latest_label.confidence if latest_label else 0.0

    if 'model' not in session:
        return initializeAL(form, .7)

    elif session['queue'] == [] and session['labels'] == []:
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []:
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return render_template('label.html', form=form, confidence=latest_confidence)

@app.route("/intermediate.html",methods=['GET'])
@login_required
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    return render_template('intermediate.html')

@app.route("/final.html",methods=['GET'])
@login_required
def final():
    """
    Operates the final(final.html) web page.
    """
    return render_template('final.html')

@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>",methods=['GET'])
@login_required
def feedback(h_list,u_list,h_conf_list,u_conf_list):
    """
    Operates the feedback(feedback.html) web page.
    """
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)
    
    return render_template('feedback.html', healthy_list = h_feedback_result, unhealthy_list = u_feedback_result, healthy_conf_list = h_conf_result, unhealthy_conf_list = u_conf_result, h_list_length = h_length, u_list_length = u_length)

@app.route('/analytics')
@login_required
def analytics():
    user_accuracy, user_total, user_correct, user_incorrect_images = calculate_accuracy("user", False, return_counts=True)
    user_model_accuracy, user_model_total, user_model_correct, user_model_incorrect_images = calculate_accuracy("user", True, return_counts=True)
    all_user_accuracy, all_user_total, all_user_correct, all_user_incorrect_images = calculate_accuracy("all", False, return_counts=True)
    all_model_accuracy, all_model_total, all_model_correct, all_model_incorrect_images = calculate_accuracy("all", True, return_counts=True)

    top_mislabeled = (
    db.session.query(Image.filename, func.count().label('count'))
    .join(Label, Image.id == Label.image_id)
    .filter(Label.model == False, Image.label.isnot(None))
    .group_by(Image.filename)
    .order_by(func.count().desc())
    .limit(3)
    .all())

    top_mislabeled_images = [{'filename': img[0], 'count': img[1]} for img in top_mislabeled]

    return render_template('analytics.html', 
                            user_accuracy=user_accuracy, 
                            user_total=user_total, 
                            user_correct=user_correct, 
                            user_incorrect_images=user_incorrect_images,
                            user_model_accuracy=user_model_accuracy,
                            user_model_total=user_model_total,    
                            user_model_correct=user_model_correct,
                            user_model_incorrect_images=user_model_incorrect_images,
                            all_user_accuracy=all_user_accuracy,
                            all_user_total=all_user_total,
                            all_user_correct=all_user_correct,
                            all_user_incorrect_images=all_user_incorrect_images,
                            all_model_accuracy=all_model_accuracy,
                            all_model_total=all_model_total,
                            all_model_correct=all_model_correct,
                            all_model_incorrect_images=all_model_incorrect_images, 
                            top_mislabeled_images=top_mislabeled_images)



#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)