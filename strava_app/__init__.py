import os
from flask import Flask

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile('config.py', silent=True)

    @app.route('/hello')
    def hello():
        return "Hello, World"

    from . import main_views
    app.register_blueprint(main_views.bp)
    return app