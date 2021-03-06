

from flask_restful import Resource, reqparse
from flask import jsonify

from common.utils import parse_json
from models.predict_model import Model

model = Model()
parser = reqparse.RequestParser()
parser.add_argument("json_data")


class Model(Resource):
    def get(self):
        return jsonify("test!!")


class Predict(Resource):
    def post(self):

        args = parser.parse_args()
        raw_json = args["json_data"]
        prediction = model.predict(parse_json(raw_json))
        probabilities = model.predict_probability(parse_json(raw_json))
        approve_loan_probability = probabilities[0][prediction[0]]

        # FIXME - find how to send it correct
        return jsonify({"prediction": prediction[0],
                        "probability": approve_loan_probability})
