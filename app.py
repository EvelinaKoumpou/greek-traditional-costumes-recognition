from flask import Flask, render_template, request
from PIL import Image

from predict import predict_image

app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recognition", methods=["GET", "POST"])
def recognition():

    results = None

    if request.method == "POST":

        uploaded_file = request.files.get("image")

        if uploaded_file and uploaded_file.filename != "":

            image = Image.open(uploaded_file)

            results = predict_image(image)

    return render_template(
        "recognition.html",
        results=results
    )


@app.route("/regions")
def regions():
    return render_template("regions.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)