from flask import Flask , render_template , flash , request , jsonify
from Text_summary import generate_summary



app = Flask(__name__)

@app.route("/text_summary" , methods = ["POST"])
def text_summary():

    if(request.method == 'POST'):
        
        data = request.get_json()
        json_data = generate_summary(data['article'])
        
        return json_data


if __name__ == "__main__":
    app.run(host="0.0.0.0",port = 3333,debug=True)