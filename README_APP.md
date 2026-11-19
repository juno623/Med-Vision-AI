# Med Vision AI - web app

Doctor uploads a handwritten prescription -> AI reads it (preprocessing, TrOCR with beam search, medicine matching) ->
doctor reviews, edits and approves -> only verified prescriptions reach the patient's history.

## 1. Install (Windows, in your activated venv, from the project root)
```
pip install -r requirements_app.txt
```
Extract this zip INTO your project folder so that `app\`, `run.py`, `config.py`, `seed.py` sit next to
`preprocessing\`, `ocr\`, `matching\` and `data\`. Check with:  `dir app`  and  `dir matching`

## 2. Point the app at your trained model
The app looks for the model in `trocr-rxhandbd-checkpoints\final_model` (your v1 model). To use another one:
```
set MODEL_DIR=D:\PROJECT1.O\trocr-rxhandbd-v3-checkpoints\final_model
```
If the model cannot be loaded the app still runs in a clearly labelled DEMO mode (yellow banner, placeholder readings).

## 3. Create the database and demo data
```
python seed.py
```
Creates the tables, the medicine reference list (from `data\RxHandBD-ML` labels) and three accounts:

| Role | Email | Password |
|---|---|---|
| Admin | admin@medvision.test | Admin@123 |
| Doctor | doctor@medvision.test | Doctor@123 |
| Patient | patient@medvision.test | Patient@123 |

`python seed.py --train-only` builds the medicine list from training labels only (use it when measuring accuracy).

## 3b. (Recommended) Widen the medicine list
Only about 64% of the medicines in the test images appear in the training labels, so the matcher can never find the rest.
Add a real brand-name list of Bangladeshi medicines, for example the Mendeley dataset
"Medicinal Products in Bangladesh" (DOI 10.17632/zhtvkny53n, cite it in your report):
```
python seed.py --extra "D:\path\to\medicines.csv"
python ocr/evaluate_with_matching.py --data_dir data/RxHandBD-ML --model_dir <final_model> --extra_vocab "D:\path\to\medicines.csv"
```
The brand-name column is found automatically; if it fails the error lists the headers, then add `--extra-column "Header"`
(`--extra_column` for the evaluation script). Names the dataset really uses get a small score bonus, so the big list adds
coverage without drowning the common names. If you changed the database layout (this version added a column), delete
`instance\medvision.db` first, then run `seed.py` again.

## 4. Run
```
python run.py
```
Open http://127.0.0.1:5000 . The first start takes ~15 s while the model loads.

## 5. Five-minute demo script
1. Log in as the doctor, search `patient@medvision.test`, open the patient.
2. Upload a prescription image. Show the AI reading and the top-3 medicine suggestions.
3. Fix one reading using a suggestion, drag a box on the image to read one more word, fill in dosage details.
4. Approve. Log out, log in as the patient, show the verified prescription in the history.
5. Log in as admin to show users, the medicine database and the audit log.

Images up to 700 px (like RxHandBD word crops) are read as one region. Bigger photos are deskewed and split into
regions (needs `easyocr`; without it the whole image is one region and you drag boxes yourself).

## 6. Measure accuracy and run the tests
```
python ocr\evaluate_with_matching.py --data_dir data\RxHandBD-ML --model_dir <path to final_model>
python -m unittest tests.test_app -v
```

## Notes for the report / viva
* Backend: Flask + SQLAlchemy. SQLite by default; set `DATABASE_URL=mysql+pymysql://user:pass@localhost/medvision` for MySQL.
* Security: hashed passwords, role-based access, CSRF tokens, upload validation, audit log. Patients only see their own
  verified prescriptions. Not production-ready (no HTTPS, rate limiting or email verification).
* The AI assists; nothing is stored in a patient's record until a doctor approves it.
* The demo medicine list includes test-set names. For honest accuracy numbers use the evaluation script (train-only list).
