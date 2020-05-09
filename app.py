import os
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory, render_template, request, make_response, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super secret'

def load_DAIN():
    # Let the magic happen
    from DAIN.DAIN import DAIN
    module = DAIN()
    
    # load the weights online
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url("http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth")
    module.load_state_dict(state_dict)

    return module

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def Home():
    if request.method == "POST":

        file = request.files["file"]

        print("File uploaded")
        print(file)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            res = make_response(jsonify({"message": "File uploaded"}), 200)
            return res
        else:
            res = make_response(jsonify({"message": "Invalid Video"}), 400)
            return res

    return render_template("home.html")


@app.route("/download")
def Download():
    return render_template("download.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.debug = True
    app.run()