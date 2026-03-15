import sys
import os
import streamlit as st


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vector_store.search_with_ranker import MultimodalSearchEngine


st.set_page_config(
    page_title="Multimodal Search Engine",
    layout="wide"
)

st.title("🔍 Multimodal Image Search Engine")
st.markdown(
    "Search images using **natural language** powered by **CLIP + FAISS + Learning-to-Rank**"
)

@st.cache_resource
def load_engine():
    return MultimodalSearchEngine()

engine = load_engine()

query = st.text_input(
    "Enter your search query:",
    placeholder="e.g. sunset beach, forest nature, mountain landscape"
)

top_k = st.slider("Number of results", 1, 10, 5)


if st.button("🔍 Search") and query.strip():
    with st.spinner("Searching..."):
        results = engine.search(query, top_k=top_k)

    st.subheader("🏆 Ranked Results")

    if not results:
        st.warning("No results found")
    else:
        for i, r in enumerate(results, 1):
            col1, col2 = st.columns([1, 3])

            # ---------------- IMAGE ----------------
            with col1:
                img = r.get("image")

                if img:
                    
                    if img.startswith("http"):
                        st.image(img, width="stretch")   # ✅ UPDATED
                    elif os.path.exists(img):
                        st.image(img, width="stretch")   # ✅ UPDATED
                    else:
                        st.warning("🖼️ Image file not found")
                        st.code(img)
                else:
                    st.info("🖼️ No image available")

            
            with col2:
                st.markdown(f"### #{i}")
                st.markdown(f"**🧠 Rank score:** `{r['rank_score']}`")
                st.markdown(f"**🔗 CLIP similarity:** `{r['clip_similarity']}`")
                st.markdown(
                    f"**📝 Description:** {r['description'] or 'No description'}"
                )

            st.divider()

elif not query.strip():
    st.info("👆 Enter a query and click Search")
