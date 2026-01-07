"""Flask application factory and entry point."""
from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    from src.routes.main import main_bp
    from src.routes.prediction import prediction_bp
    from src.routes.comparison import comparison_bp
    from src.routes.redirects import redirects_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(comparison_bp)
    app.register_blueprint(redirects_bp)
    
    return app


app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
