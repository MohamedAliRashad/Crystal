import os

from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename
from subprocess import Popen

UPLOAD_FOLDER = "./uploads"
DOWNLOAD_FOLDER = "./downloads"
ALLOWED_EXTENSIONS = {"mp4", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "super secret"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def Home():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    if not os.path.isdir(DOWNLOAD_FOLDER):
        os.mkdir(DOWNLOAD_FOLDER)
    
    if request.method == "POST":

        file = request.files["file"]

        print("File uploaded")
        print(file)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            res = make_response(jsonify({"message": "File uploaded"}), 200)
            process = Popen(["python3", "main.py", os.path.join(app.config["UPLOAD_FOLDER"], filename)])
            return res
        else:
            res = make_response(jsonify({"message": "Invalid Video"}), 400)
            return res

    return render_template("home.html")


@app.route("/tour")
def Tour():
    return render_template("tour.html")


@app.route("/upgrade")
def Upgrade():
    return render_template("upgrade.html")


@app.route("/help")
def Help():
    return render_template("help.html")


@app.route("/explore")
def Explore():
    return render_template("explore.html")


@app.route("/profile")
def Profile():
    return render_template("profile.html")


@app.route("/download")
def Download():
    name = os.listdir(os.path.join(app.root_path, app.config["DOWNLOAD_FOLDER"]))[0]
    return render_template("download.html", name=name)


@app.route("/downloads/<path:filename>", methods=["GET", "POST"])
def download_video(filename):
    downloads = os.path.join(app.root_path, app.config["DOWNLOAD_FOLDER"])
    return send_from_directory(directory=downloads, filename=filename, as_attachment=True)


if __name__ == "__main__":
    app.debug = True
    app.run()
