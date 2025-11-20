# Dota Oracle

## Getting Started

### Install

* Go to project root `cd dota-oracle`
* Make sure the conda/virtualenv environment you want to install to is active.
* Run `pip install -e .`

### Run

* Development server: `python -m doracle.app`
* Production (Linux/macOS): `pip install -e .[prod]` then `gunicorn -c gunicorn.conf.py doracle.app:app`

> Gunicorn relies on POSIX process forking and is not supported on native Windows shells. Use WSL2 or a Linux container when testing locally on Windows.

### Docker

If you prefer a portable container image, build it from the repo root so the supporting data files stay in sync:

```
docker build -t dota-oracle -f dota/dota-oracle/Dockerfile .
```

Run the container (mapping port 8000 by default):

```
docker run --rm -p 8000:8000 dota-oracle
```

Override any app settings with env vars (e.g. `-e DORACLE_PORT=5000`).

## License

[MIT License](LICENSE)
