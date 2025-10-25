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

dataset_generator_RWx.py you can use it to generate new training dataset pair (train and test)


git status
git remote -v
git fetch origin
git switch AhmetKemal
git add -u
git status
git commit -m "ENNC & libs updates; add dataset_generator_RWx.py; README edits"
git push