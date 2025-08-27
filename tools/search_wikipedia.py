try:
    import wikipedia
except ImportError:
    wikipedia = None

def search_wikipedia(query):
    """Search Wikipedia for a query and return a summary."""
    if wikipedia is None:
        return "Wikipedia module not installed. Run 'pip install wikipedia'."
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Wikipedia search error: {e}"
