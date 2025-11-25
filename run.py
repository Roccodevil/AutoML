from app import create_app, mongo

# Create the Flask app instance using the factory
app = create_app()

@app.shell_context_processor
def make_shell_context():
    """Provides the 'mongo' object to the 'flask shell'"""
    return {'mongo': mongo}

if __name__ == '__main__':
    # Run the Flask app
    # debug=True reloads the server on code changes
    # host='0.0.0.0' makes it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000)