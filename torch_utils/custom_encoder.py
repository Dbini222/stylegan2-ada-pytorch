import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            # Convert non-serializable objects to None or custom string
            return f"Non-serializable object of type {type(obj)}"


