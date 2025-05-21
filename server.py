#!/usr/bin/env python3

import os
from flask import Flask, jsonify, request

import json
from predict_hw import predict


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def flask_app():
    app = Flask(__name__)


    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up - nice job! \n \n'

    @app.route('/predict_mpg', methods=['POST'])
    def start():
        to_predict = request.json

        print(to_predict)
        pred = predict(to_predict)
        return jsonify({"predict mpg":pred})
    return app

if __name__ == '__main__':
    app = flask_app()
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host="0.0.0.0", port=port)   


