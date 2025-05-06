\
# wsgi.py
from main_app import app

if __name__ == "__main__":
    # This part is for local execution if you run `python wsgi.py`
    # Render will use Gunicorn or a similar server to import `app` from main_app.py
    print("Starting Flask app via wsgi.py for local testing...")
    app.run(host='0.0.0.0', port=5001, debug=False) # Match port with main_app.py for local
