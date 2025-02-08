# gunicorn.conf.py

# Server socket binding
bind = '0.0.0.0:30000'

# Worker configuration
workers = 1
threads = 5

# SSL/TLS configuration
certfile = 'cert.pem'
keyfile = 'key.pem'

# Logging configuration
accesslog = '-'  # '-' means log to stdout
errorlog = '-'   # '-' means log to stderr

# The application to load
wsgi_app = 'app:app'
