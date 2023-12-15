# Deployment

## Docker-Compose
There is a docker image available with this repository. that is road-detection.
git clone this repo. and cd into deployment and run docker-compose up.
open http://localhost:7860/ in you browser to see the app

## Docker
you can run the following command. This will download the image and deploy it. open http://localhost:7860/ in you browser to see the app.

"docker run -p 7860:7860 -e SHARE=True ghcr.io/balnarendrasapa/road-detection:latest"

## Python Virtual Environment
cd into deployment directory. and run "python -m venv .venv" to create a virtual environment.
run "pip install -r requirements.txt"
run "python app.py"
open http://localhost:7860/ in you browser to see the app

[Youtube Presentation](https://youtu.be/bnyA-d6lZi8)