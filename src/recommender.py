from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.config import (
    SIMILARITY_THRESHOLD,
    TOP_N
)


def get_recommandation(
    query: str,
    vectorizer,
    tfidf_matrix,
    df: pd.DataFrame
):

    query_vector = vectorizer.transform(
        [query.lower()]
    )

    sim_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    max_sim = float(sim_scores.max())

    print(
        f"Debug Query: '{query}' "
        f"| Max Similarity: {max_sim:.4f}"
    )

    # =====================================================
    # DATABASE MATCH
    # =====================================================

    if max_sim >= SIMILARITY_THRESHOLD:

        top_indices = sim_scores.argsort()[::-1]

        filtered_indices = [

            i for i in top_indices

            if sim_scores[i]
            >= SIMILARITY_THRESHOLD
        ]

        top_df = df.iloc[
            filtered_indices[:TOP_N]
        ].copy()

        source = "database"

    # =====================================================
    # OPENROUTER FALLBACK
    # =====================================================

    else:

        top_df = None

        source = "openrouter_ai"

    return top_df, source, max_sim