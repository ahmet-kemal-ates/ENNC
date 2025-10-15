# ENNC
Synthesis of a data-driven ESS model by using real cell tests data

Works on Python 3.11

python -m venv venv

.\venv\Scripts\Activate.ps1

python --version (got to be 3.11.x)

pip install -r requirements.txt

cd "ENNC\Ststem Identification"

python main_Train.py

log files will be generated automatically during every run under the "ENNC/Ststem Identification/Models" directory with timestamps
