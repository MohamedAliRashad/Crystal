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

UPLOAD_FOLDER = "./uploads"
FRAMES_FOLDER = "./frames"
ALLOWED_EXTENSIONS = {"mp4", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "super secret"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def Home():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    if request.method == "POST":

        file = request.files["file"]

        print("File uploaded")
        print(file)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            res = make_response(jsonify({"message": "File uploaded"}), 200)
            return res

        else:
            res = make_response(jsonify({"message": "Invalid Video"}), 400)
            return res

    return render_template("home.html")


@app.route("/download")
def Download():
    return render_template("download.html")


@app.route("/loading/<filename>")
def loading(filename=None):
    if(not filename):
        return "There's no file uploaded"
    return render_template("loading.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
