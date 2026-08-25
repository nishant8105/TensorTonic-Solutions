import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    cosine_similarity = np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
)

    if np.isnan(cosine_similarity):
        cosine_similarity = 0

    return float(cosine_similarity)
    pass