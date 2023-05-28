from flask import Flask,render_template,request
import pickle
import os
from PIL import Image
model = pickle.load(open("irismdl.pkl",'rb'))
app = Flask(__name__)
@app.route("/", methods=['GET','POST'])
def start():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        i = request.form['sl']
        k = request.form['sw']
        j = request.form['pl']
        l = request.form['pw']
        features = [[i, j, k, l]]
        pred = model.predict(features)[0]
        pdn = ''
        if int(pred) == 0:
            pdn = 'setosa'
        elif int(pred) == 1:
            pdn = 'versicolor'
        elif int(pred) == 2:
            pdn = 'virginica'
        file = os.path.join('static', f'{pdn}.jpeg')
        return render_template('result.html', data = pdn,image = file)

if __name__ == '__main__':
    app.run(debug=True)

