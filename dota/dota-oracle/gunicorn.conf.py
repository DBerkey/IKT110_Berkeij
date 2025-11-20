import multiprocessing
import os

# Bind can be overridden with DORACLE_GUNICORN_BIND, e.g. "0.0.0.0:8080" for containers.
bind = os.getenv("DORACLE_GUNICORN_BIND", "0.0.0.0:5000")

# Workers default to half the available CPUs but at least one.
_cpu_count = multiprocessing.cpu_count() or 1
workers = int(os.getenv("WEB_CONCURRENCY", str(max(1, _cpu_count // 2))))

# Threads improve latency for IO heavy workloads.
threads = int(os.getenv("GUNICORN_THREADS", "2"))

timeout = int(os.getenv("GUNICORN_TIMEOUT", "30"))
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")

# Preload speeds worker fork and keeps model weights shared when supported.
preload_app = True
