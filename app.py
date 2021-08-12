from flask import Flask, render_template, request
import pickle


BOW = pickle.load(open('bag_of_words.pkl', 'rb'))
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))

app = Flask(__name__)
# @app.route('/')
# def index():
#     value = 'hello world'
#     context =value
#     print(type(context))
#     return render_template('index.html', context= type(context))

@app.route('/check/', methods=['GET', 'POST'])
def check():
    if request.method == 'POST':

        message = request.form['message']
        data = [message]
        vector = BOW.transform(data).toarray()
        prediction = model.predict(vector)
        return render_template('home.html', value=prediction)
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)