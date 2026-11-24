"""Flask application factory."""

from flask import Flask, send_file

from .paths import STATIC_DIR, DATA_ROOT


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    from .routes.assets  import bp as assets_bp
    from .routes.config  import bp as config_bp
    from .routes.images  import bp as images_bp
    from .routes.prints  import bp as prints_bp
    from .routes.zines   import bp as zines_bp

    for bp in (assets_bp, config_bp, images_bp, prints_bp, zines_bp):
        app.register_blueprint(bp)

    @app.route("/")
    def index():
        return send_file(STATIC_DIR / "index.html")

    print(f"\nDE BRUITS  →  http://localhost:5000")
    print(f"  Data root : {DATA_ROOT}\n")

    return app


if __name__ == "__main__":
    create_app().run(debug=False, port=5000)
