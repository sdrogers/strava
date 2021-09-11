import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile('config.py', silent=True)
    csrf = CSRFProtect(app)

    SECRET_KEY = os.urandom(32)
    app.config['SECRET_KEY'] = SECRET_KEY
    
    @app.route('/hello')
    def hello():
        return "Hello, World"

    from . import main_views, app_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(app_views.av)
    return app