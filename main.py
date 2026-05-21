import os
import traceback

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.data_loader import load_data
from src.data_preprocessing import create_combined_text
from src.vectorizer import create_vectorizer
from src.recommender import get_recommandation
from src.openrouter_helper import get_openrouter_response

app = FastAPI(title="Medicine Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
vectorizer = None
tfidf_matrix = None


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    global df, vectorizer, tfidf_matrix

    try:

        df_raw = load_data()

        df = create_combined_text(df_raw)

        vectorizer = create_vectorizer(df)

        tfidf_matrix = vectorizer.transform(
            df['combined_text']
        )

        print(f"✅ Loaded {len(df)} medicines")

    except Exception as e:

        print(f"❌ Startup Error: {e}")


# =========================================================
# HOME
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    html_path = os.path.join(
        os.path.dirname(__file__),
        "app",
        "templates",
        "index.html"
    )

    try:

        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    except Exception:

        return HTMLResponse(
            "<h1>index.html not found</h1>"
        )


# =========================================================
# RECOMMEND
# =========================================================

@app.post("/recommend")
async def recommend(
    symptoms: str = Form(...)
):

    try:

        top_df, source, max_sim = get_recommandation(
            symptoms,
            vectorizer,
            tfidf_matrix,
            df
        )

        medicines = []

        ai_response = {}

        # =================================================
        # DATABASE MATCH
        # =================================================

        if top_df is not None and not top_df.empty:

            source = "database"

            for _, row in top_df.iterrows():

                medicines.append({

                    "name": str(
                        row.get('NAME', '')
                    ),

                    "contains": str(
                        row.get('CONTAINS', '')
                    ),

                    "uses": str(
                        row.get('USES', '')
                    ),

                    "benefits": str(
                        row.get('BENEFITS', '')
                    ),

                    "side_effects": str(
                        row.get('SIDE_EFFECT', '')
                    )
                })

            print(
                f"✅ Database Match | Similarity: {max_sim:.3f}"
            )

        # =================================================
        # OPENROUTER FALLBACK
        # =================================================

        else:

            source = "openrouter_ai"

            ai_response = get_openrouter_response(
                symptoms
            )

            print(
                f"🌐 OpenRouter Used | Similarity: {max_sim:.3f}"
            )

        return {

            "symptoms": symptoms,

            "source": source,

            "similarity": round(
                float(max_sim),
                3
            ),

            "medicines": medicines,

            "ai_response": ai_response
        }

    except Exception as e:

        print(
            "🔥 ERROR:",
            traceback.format_exc()
        )

        return JSONResponse(
            {
                "error": str(e)
            },
            status_code=500
        )

