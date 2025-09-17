import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_mail import Mail, Message

load_dotenv()

app = Flask(__name__)

# Configuration SMTP LaPoste
app.config["MAIL_SERVER"] = "smtp.laposte.net"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

mail = Mail(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/articles")
def articles():
    return render_template("articles.html")

@app.route("/a_propos")
def a_propos():
    return render_template("a_propos.html")

@app.route("/mentions_legales")
def mentions_legales():
    return render_template("mentions_legales.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        nom = request.form.get("nom")
        email = request.form.get("email")
        message = request.form.get("message")
        
        msg = Message(
            subject="Nouveau message du site Molynx",
            sender=app.config["MAIL_USERNAME"],
            recipients=["maxime.infuso@laposte.net"],
            body=f"Nom : {nom}\nEmail : {email}\n\nMessage :\n{message}"
        )

        print("=== Nouveau message reçu ===")
        print(f"Nom     : {nom}")
        print(f"Email   : {email}")
        print(f"Message : {message}")
        print("============================")
        
        mail.send(msg)
        
        return render_template("index.html")

    # si on arrive sur /contact en GET (rare dans ton cas), on peut rediriger
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
