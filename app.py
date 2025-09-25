import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask import send_from_directory, url_for, make_response, redirect
from flask_mail import Mail, Message

from dashboard.dashboard_crypto.app import init_dashboard


load_dotenv()

server = Flask(__name__)

# Dashboards initialisation
dash_app = init_dashboard(server) # Dashboard-crypto

# Configuration SMTP LaPoste
server.config["MAIL_SERVER"] = "pro3.mail.ovh.net"
server.config["MAIL_PORT"] = 587
server.config["MAIL_USE_TLS"] = True
server.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
server.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
server.config['PREFERRED_URL_SCHEME'] = 'https'

mail = Mail(server)

@server.route("/")
def home():
    return render_template("index.html")

@server.route("/services")
def services():
    return render_template("services.html")

@server.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

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
            recipients=["contact@molynx.fr"],
            body=f"Nom : {nom}\nEmail : {email}\n\nMessage :\n{message}"
        )

        print("=== Nouveau message reçu ===")
        print(f"Nom     : {nom}")
        print(f"Email   : {email}")
        print(f"Message : {message}")
        print("============================")
        
        mail.send(msg)
        
        return redirect(url_for("contact"))

    # si on arrive sur /contact en GET (rare dans ton cas), on peut rediriger
    return render_template("index.html")

@server.route('/robots.txt')
def serve_robots():
    return send_from_directory(server.static_folder, 'robots.txt')


@server.route('/sitemap.xml')
def sitemap():
    pages = []

    # Liste des noms de routes statiques que tu veux inclure
    static_endpoints = [
        'home',        # ta page d’accueil
        'services',
        'portfolio',
        'articles',
        'a_propos',
        'mentions_legales',
        'contact'
    ]

    for ep in static_endpoints:
        try:
            url = url_for(ep, _external=True, _scheme='https')
        except Exception as e:
            # Si l’endpoint n’existe pas, on saute
            continue
        pages.append({
            'loc': url,
        })

    xml = render_template('sitemap_template.xml', pages=pages)
    response = make_response(xml)
    response.headers['Content-Type'] = 'application/xml'
    return response

if __name__ == "__main__":
    server.run(debug=True)
