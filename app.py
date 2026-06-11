from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import send_file

from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
import joblib

app = Flask(__name__)

# ==========================
# Database Configuration
# ==========================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///loan.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==========================
# Load ML Model
# ==========================
model = joblib.load("loan_model.pkl")

# ==========================
# Applicant Table
# ==========================
class Applicant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    income = db.Column(db.Integer)
    loan_amount = db.Column(db.Integer)
    credit_history = db.Column(db.String(20))
    status = db.Column(db.String(20))

# ==========================
# Create Database
# ==========================
with app.app_context():
    db.create_all()

# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("home.html")

# ==========================
# Login Page
# ==========================
@app.route("/login")
def login():
    return render_template("login.html")

# ==========================
# Dashboard
# ==========================
@app.route("/dashboard")
def dashboard():

    total_applications = Applicant.query.count()

    approved_loans = Applicant.query.filter_by(
        status="Approved"
    ).count()

    rejected_loans = Applicant.query.filter_by(
        status="Rejected"
    ).count()

    high_risk_cases = rejected_loans

    return render_template(
        "dashboard.html",
        total_applications=total_applications,
        approved_loans=approved_loans,
        rejected_loans=rejected_loans,
        high_risk_cases=high_risk_cases
    )

# ==========================
# Add Applicant
# ==========================
@app.route("/add-applicant", methods=["GET", "POST"])
def add_applicant():

    if request.method == "POST":

        name = request.form["name"]
        income = int(request.form["income"])
        loan_amount = int(request.form["loan_amount"])
        credit_history = request.form["credit_history"]

        # Convert Good/Bad into 1/0
        credit = 1 if credit_history == "Good" else 0

        # ML Prediction
        prediction = model.predict([
            [income, loan_amount, credit]
        ])

        probability = model.predict_proba([
            [income, loan_amount, credit]
        ])[0]

        approval_probability = round(
            probability[1] * 100,
            2
        )

        if prediction[0] == 1:
            status = "Approved"
        else:
            status = "Rejected"

        # Risk Level
        if approval_probability >= 80:
            risk_level = "LOW"

        elif approval_probability >= 50:
            risk_level = "MEDIUM"

        else:
            risk_level = "HIGH"

        applicant = Applicant(
            name=name,
            income=income,
            loan_amount=loan_amount,
            credit_history=credit_history,
            status=status
        )

        db.session.add(applicant)
        db.session.commit()

        return render_template(
            "result.html",
            name=name,
            status=status,
            approval_probability=approval_probability,
            risk_level=risk_level
        )

    return render_template("add_applicant.html")

# ==========================
# View Applicants + Search
# ==========================
@app.route("/applicants")
def applicants():

    search = request.args.get("search")

    if search:

        all_applicants = Applicant.query.filter(
            Applicant.name.contains(search)
        ).all()

    else:

        all_applicants = Applicant.query.all()

    return render_template(
        "applicants.html",
        applicants=all_applicants
    )

# ==========================
# Delete Applicant
# ==========================
@app.route("/delete/<int:id>")
def delete_applicant(id):

    applicant = Applicant.query.get_or_404(id)

    db.session.delete(applicant)

    db.session.commit()

    return redirect("/applicants")

# ==========================
# Edit Applicant
# ==========================
@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_applicant(id):

    applicant = Applicant.query.get_or_404(id)

    if request.method == "POST":

        applicant.name = request.form["name"]
        applicant.income = int(request.form["income"])
        applicant.loan_amount = int(request.form["loan_amount"])
        applicant.credit_history = request.form["credit_history"]

        db.session.commit()

        return redirect("/applicants")

    return render_template(
        "edit_applicant.html",
        applicant=applicant
    )

# ==========================
# Run App
# ==========================
@app.route("/report/<int:id>")
def generate_report(id):

    applicant = Applicant.query.get_or_404(id)

    pdf_file = f"loan_report_{id}.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("SmartLoan AI Report", styles["Title"])
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            f"Applicant Name: {applicant.name}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Income: {applicant.income}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Loan Amount: {applicant.loan_amount}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Credit History: {applicant.credit_history}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Status: {applicant.status}",
            styles["Normal"]
        )
    )

    doc.build(content)

    return send_file(
        pdf_file,
        as_attachment=True
    )
@app.route("/model-info")
def model_info():

    return render_template(
        "model_info.html"
    )
if __name__ == "__main__":
    app.run(debug=True)