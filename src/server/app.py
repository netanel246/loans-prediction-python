from flask import Flask
from flask_restful import Api
from server.resources.model import Model, Predict

app = Flask(__name__)
api = Api(app)


api.add_resource(Model, '/model/get_params')
api.add_resource(Predict, '/model/predict')


if __name__ == "__main__":
    app.run(debug=True, port=8089)
