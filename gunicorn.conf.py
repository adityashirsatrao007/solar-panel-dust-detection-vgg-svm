import os

bind = "0.0.0.0:5001"
workers = 1
timeout = 120
# Log to files aditya can create (the service now runs as a non-root user,
# so it must not depend on the pre-existing root-owned log files).
_logdir = os.path.join(os.path.dirname(__file__), "logs")
accesslog = os.path.join(_logdir, "gunicorn_access.log")
errorlog = os.path.join(_logdir, "gunicorn_error.log")
capture_output = True
