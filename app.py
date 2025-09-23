import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_mail import Mail, Message

from dashboard.dashboard_crypto.app import init_dashboard


load_dotenv()

server = Flask(__name__)

# Dashboards initialisation
dash_app = init_dashboard(server) # Dashboard-crypto

# Configuration SMTP LaPoste
server.config["MAIL_SERVER"] = "smtp.laposte.net"
server.config["MAIL_PORT"] = 587
server.config["MAIL_USE_TLS"] = True
server.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
server.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")

mail = Mail(server)

@server.route("/")
def home():
    return render_template("index.html")

@server.route("/services")
def services():
    return render_template("services.html")

@server.route("/articles")
def articles():
    return render_template("articles.html")

@server.route("/a_propos")
def a_propos():
    return render_template("a_propos.html")

@server.route("/mentions_legales")
def mentions_legales():
    return render_template("mentions_legales.html")

@server.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        nom = request.form.get("nom")
        email = request.form.get("email")
        message = request.form.get("message")
        
        msg = Message(
            subject="Nouveau message du site Molynx",
            sender=server.config["MAIL_USERNAME"],
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
    server.run(debug=True)
