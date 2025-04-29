# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
from flask import Flask, g
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from app import app, db
from app.models import User, Label, Image
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sqlalchemy import desc
from sklearn.ensemble import RandomForestClassifier
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
    
    # Ensure user is logged in and user_id is available
    if not current_user.is_authenticated:
        return redirect(url_for('login'))

    # Retrieve user_id from the logged-in user (current_user)
    user_id = current_user.id
    
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
    
    #loops through session['session'] and adds the image and label to the database
    for filename, label in session['sample']:
        new_image = Image(filename=filename, user_id=user_id, label=label)
        db.session.add(new_image)
        db.session.commit()
        new_label = Label(text=label, image_id=new_image.id, user_id=user_id, confidence=confidence_score)
        db.session.add(new_label)
        db.session.commit()
        
    if session['train'] != None:
        session['train'] = session.get('train', [])
        session['train'].extend(session['sample'])
    else:
        session['train'] = session['sample']

    data = getData()
    ml_model, train_img_names = createMLModel(data)
    
    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic, blight_user = blight_pic, healthNum_user = len(health_pic), blightNum_user = len(blight_pic))
    else:
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
        return render_template('final.html', form = form, confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)

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
            login_user(user)
            if user.session:
                session.update(user.session) 
            return redirect(url_for('label'))
        else:
            flash('Login failed. Please check your email and password and try again.', 'danger')
    
    return render_template('login.html')  # Render the login form if it's a GET request

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def calculate_user_accuracy(return_counts=False):
    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, 'data', 'csvOut.xlsx')
    df = pd.read_excel(file_path, usecols=[0, 16], header=None)
    df.columns = ['filename', 'true_label']
    ground_truth = dict(zip(df['filename'], df['true_label']))

    user_images = Image.query.filter_by(user_id=current_user.id).all()

    total = 0
    correct = 0
    incorrect_images = []

    for img in user_images:
        true_label = ground_truth.get(img.filename)
        if true_label:
            total += 1
            if img.label.upper() == str(true_label).upper():
                correct += 1
            else:
                incorrect_images.append((img.filename, img.label, true_label))

    accuracy = round((correct / total) * 100, 2) if total > 0 else 0
    return (accuracy, total, correct, incorrect_images) if return_counts else accuracy

@app.before_request
def before_request():
    if current_user.is_authenticated:
        latest_label = Label.query.filter_by(user_id=current_user.id).order_by(desc(Label.id)).first()
        g.latest_confidence = latest_label.confidence if latest_label else 0.0


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
    # Query images based on the current user and their labels
    healthy_plants = Image.query.filter_by(user_id=current_user.id, label='H').all()
    unhealthy_plants = Image.query.filter_by(user_id=current_user.id, label='B').all()

    # Count the healthy and unhealthy images
    healthy_count = len(healthy_plants)
    unhealthy_count = len(unhealthy_plants)

    # Prepare the image URLs and pass them to the template
    healthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}"
    } for plant in healthy_plants]

    unhealthy_plants_data = [{
        'image_url': f"https://agro-ai-maize.s3.us-east-2.amazonaws.com/images_compressed/{plant.filename}",
        'name': plant.filename,
        'details_url': f"/image/{plant.id}"
    } for plant in unhealthy_plants]

    return render_template('saved.html', 
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

    return renderLabel(form)

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
    accuracy, total, correct, incorrect_images = calculate_user_accuracy(return_counts=True)

    return render_template('analytics.html', accuracy=accuracy, total=total, correct=correct, incorrect=incorrect_images)

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)

# after every request, save back into the User.session JSON column
@app.teardown_request
def save_user_session(exc):
    if current_user.is_authenticated:
        current_user.session = dict(session)
        db.session.commit()
