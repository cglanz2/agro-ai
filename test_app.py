import os
import sys
import pytest

# Ensure the project root is in the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from app.web import app, db
from app.models import User
from werkzeug.security import generate_password_hash

# Configure the app for testing: use an in-memory SQLite database
app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
app.config['SECRET_KEY'] = 'test_secret_key'

@pytest.fixture
def client():
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

def register(client, email, password):
    """Helper function to register a new user."""
    return client.post('/register', data={
        'email': email,
        'password': password,
        'retype-password': password
    }, follow_redirects=True)

def login(client, email, password):
    """Helper function to log in a user."""
    return client.post('/login', data={
        'email': email,
        'password': password
    }, follow_redirects=True)

def logout(client):
    """Helper function to log out the current user."""
    return client.get('/logout', follow_redirects=True)

def test_home_not_logged_in(client):
    """
    When not logged in, accessing the home route ('/') should
    render the public index page.
    """
    response = client.get('/')
    # Check that the response status code is 200 and contains the DOCTYPE declaration.
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data

def test_register_and_login(client):
    """
    Test that a user can register and then log in.
    """
    email = "testuser@example.com"
    password = "password123"
    
    # Register the new user.
    response = register(client, email, password)
    # Check for a success message or a hint to log in.
    assert b'Registration successful' in response.data or b'log in' in response.data
    
    # Log in with the new user.
    response = login(client, email, password)
    # The /login route should redirect to the label page on success.
    assert response.status_code == 200
    # Check for some keyword that appears in label.html (adjust as needed).
    assert b'Label' in response.data or b'confidence' in response.data

def test_protected_routes(client):
    """
    Ensure that routes decorated with @login_required (profile, label, saved, final, intermediate, feedback)
    redirect to the login page when the user is not authenticated.
    """
    protected_routes = [
        '/profile',
        '/label.html',
        '/saved',
        '/final.html',
        '/intermediate.html',
        '/feedback/a,b/a,b/a,b/a,b'  # dummy parameters for feedback route
    ]
    for route in protected_routes:
        response = client.get(route, follow_redirects=True)
        # The login page should appear for unauthorized access.
        assert b'Login' in response.data

def test_logout(client):
    """
    Test that logging out actually ends the user session.
    """
    email = "test2@example.com"
    password = "password456"
    
    # Register and log in the user.
    register(client, email, password)
    login(client, email, password)
    
    # Logout the user.
    response = logout(client)
    # After logging out, accessing a protected route should redirect to login.
    response = client.get('/profile', follow_redirects=True)
    assert b'Login' in response.data
