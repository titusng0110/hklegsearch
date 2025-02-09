# gunicorn.conf.py

# Server socket binding
bind = '0.0.0.0:30000'

# Worker configuration
workers = 1
threads = 5

# Logging configuration
accesslog = 'access.log'  # '-' means log to stdout
errorlog = 'error.log'   # '-' means log to stderr

# The application to load
wsgi_app = 'app:app'
