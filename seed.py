"""Create demo accounts, the medicine reference database and one verified prescription.

    python seed.py                 # medicines from RxHandBD train + test labels (best for the demo)
    python seed.py --train-only    # train labels only (use this vocabulary when measuring accuracy)
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

from app import create_app
from app.models import Doctor, HistoryEntry, Medicine, Patient, Prescription, PrescriptionItem, User, db, utcnow
from config import BASE_DIR
from matching.medicine_matcher import load_extra_names, load_vocab_from_csv, split_name_strength

FALLBACK = ["Napa", "Napa Extra", "Seclo", "Maxpro", "Ace", "Zimax", "Fexo", "Monas", "Filmet", "Ciprocin",
            "Neurobion", "Rivotril", "Losita", "Indever", "Inderen", "Econate", "Exeptim", "Nexcital", "Asynta",
            "Dexter", "Clavusef", "Dophyllin", "Delfian", "Paloxi", "Clacef", "Escilex"]
ACCOUNTS = [("admin@medvision.test", "Admin@123", "admin", "Admin"),
            ("doctor@medvision.test", "Doctor@123", "doctor", "Asha Verma"),
            ("patient@medvision.test", "Patient@123", "patient", "Rahul Sharma")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--labels", nargs="*", help="CSV files with a 'Text' column")
    ap.add_argument("--extra", help="Big external medicine list (.csv/.xlsx) to widen coverage")
    ap.add_argument("--extra-column", help="Header of the brand-name column (auto-detected if omitted)")
    args = ap.parse_args()
    data = BASE_DIR / "data" / "RxHandBD-ML"
    files = args.labels or [data / "Train_Label.csv"] + ([] if args.train_only else [data / "Test_Labels.csv"])

    names = set()
    for f in files:
        if Path(f).exists():
            for text in load_vocab_from_csv(str(f)):
                base, _ = split_name_strength(text)
                if len(base) >= 2:
                    names.add(base.title())
    source = "RxHandBD labels" if names else "built-in fallback list"
    names = names or set(FALLBACK)

    app = create_app({"OCR_MODE": "mock"})       # no need to load the model just to seed
    with app.app_context():
        added = 0
        extra = set()
        if args.extra:
            extra = {n.title() for n in load_extra_names(args.extra, args.extra_column)} - names
        for n in sorted(names):
            if not Medicine.query.filter_by(name=n).first():
                db.session.add(Medicine(name=n, common=True))
                added += 1
        for n in sorted(extra):
            if not Medicine.query.filter_by(name=n).first():
                db.session.add(Medicine(name=n, common=False))
                added += 1
        users = {}
        for email, pw, role, name in ACCOUNTS:
            u = User.query.filter_by(email=email).first()
            if not u:
                u = User(email=email, role=role, name=name)
                u.set_password(pw)
                db.session.add(u)
                db.session.flush()
                if role == "doctor":
                    db.session.add(Doctor(user_id=u.id, specialization="General Medicine"))
                if role == "patient":
                    db.session.add(Patient(user_id=u.id, dob="1998-04-12", gender="Male", phone="9000000000", allergies="Penicillin"))
            users[role] = u
        db.session.commit()
        patient, doctor = users["patient"].patient, users["doctor"].doctor
        if not patient.prescriptions:
            db.session.add(HistoryEntry(patient_id=patient.id, doctor_id=doctor.id, diagnosis="Seasonal fever",
                                        treatment="Rest, fluids, paracetamol", notes="Review if fever lasts beyond 3 days."))
            rx = Prescription(patient_id=patient.id, doctor_id=doctor.id, status="verified", ocr_mode="seed",
                              verified_at=utcnow(), notes="Sample record created by seed.py")
            rx.items = [PrescriptionItem(final_name="Napa", strength="500 mg", dosage="1 tablet", frequency="1-0-1", duration="5 days", needs_review=False),
                        PrescriptionItem(final_name="Seclo", strength="20 mg", dosage="1 capsule", frequency="1-0-0", duration="7 days", needs_review=False)]
            db.session.add(rx)
        db.session.commit()
        print(f"Medicine reference DB: {Medicine.query.count()} medicines ({added} added from {source}).")
    print("\nDemo accounts:")
    for email, pw, role, _ in ACCOUNTS:
        print(f"  {role:8} {email:26} {pw}")


if __name__ == "__main__":
    main()
