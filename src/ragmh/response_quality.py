def enhance_response_quality(response: str, query: str) -> str:
    """Post-process responses for empathy, suggestions, and safety hints."""
    word_count = len(response.split())
    if word_count < 40:
        response += "\n\nCould you tell me more about what you're experiencing? I'm here to help."
    empathy_words = ["understand", "difficult", "sorry", "challenging"]
    if not any(word in response.lower() for word in empathy_words):
        response = "I hear that you're going through something difficult. " + response
    serious_topics = ["depression", "anxiety", "trauma", "ptsd", "suicide"]
    if any(topic in query.lower() for topic in serious_topics):
        if "professional" not in response.lower() and "therapist" not in response.lower():
            response += "\n\nIf these feelings persist, consider speaking with a mental health professional who can provide personalized support."
    return response.strip()