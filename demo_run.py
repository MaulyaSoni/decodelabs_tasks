# ============================================================
#  DecodeLabs | Industrial Training Kit | Batch 2026
#  demo_run.py : Automated Demo — No typing needed
#  Purpose     : Show all 4 projects working instantly
#                Perfect for viva / presentation / submission
# ============================================================

import sys, os, time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "project1_chatbot"))
sys.path.insert(0, os.path.join(BASE_DIR, "project2_classification"))
sys.path.insert(0, os.path.join(BASE_DIR, "project3_recommendation"))
sys.path.insert(0, os.path.join(BASE_DIR, "project4_recognition"))

def sep(title=""): print("\n"+"="*60+f"\n  {title}\n"+"="*60 if title else "\n"+"="*60)
def pause(s=0.8): time.sleep(s)

# ── DEMO P1 V1 ───────────────────────────────────────────────
def demo_chatbot_v1():
    sep("DEMO: PROJECT 1 — RULE-BASED CHATBOT (v1)")
    from chatbot_v1 import sanitize, get_response
    inputs = ["hello","who are you","what is ai","what is ml","tell me a joke","motivate me","what are the projects","thank you"]
    for u in inputs:
        print(f"You  : {u}")
        print(f"DBot : {get_response(sanitize(u))}\n")
        pause(0.2)
    print("  ✓ Chatbot V1 Demo Complete")

# ── DEMO P1 V2 ───────────────────────────────────────────────
def demo_chatbot_v2():
    sep("DEMO: PROJECT 1 — HYBRID CHATBOT (v2) — Rule Layers")
    from chatbot_v2 import sanitize, get_response
    inputs = ["hey","what is deep learning","explain transformers","what is groq","what are the projects","tell me a joke","what time is it"]
    for u in inputs:
        r, src = get_response(sanitize(u), None, [])
        print(f"You  : {u}")
        print(f"DBot : {r}")
        print(f"       ↳ {src}\n")
        pause(0.2)
    print("  ✓ Chatbot V2 Demo Complete")
    print("  [NOTE] Add GROQ_API_KEY in chatbot_v2.py to unlock Llama 3.3")

# ── DEMO P2 ──────────────────────────────────────────────────
def demo_classification():
    sep("DEMO: PROJECT 2 — DATA CLASSIFICATION USING AI")
    import numpy as np
    from classifier import load_data, split_data, scale_features, train_model, evaluate_model
    X, y, cn = load_data()
    Xtr, Xte, ytr, yte = split_data(X, y)
    Xtr_s, Xte_s, sc   = scale_features(Xtr, Xte)
    model               = train_model(Xtr_s, ytr, k=5)
    preds, acc, f1      = evaluate_model(model, Xte_s, yte, cn)
    sep("DEMO PREDICTIONS — 3 Sample Flowers")
    for feat, exp in [([5.1,3.5,1.4,0.2],"setosa"),([6.0,2.9,4.5,1.5],"versicolor"),([6.7,3.0,5.2,2.3],"virginica")]:
        s  = sc.transform(np.array([feat]))
        p  = model.predict(s)[0]
        pr = model.predict_proba(s)[0]
        ok = "✓ CORRECT" if cn[p].lower()==exp else "✗ WRONG"
        print(f"  Input: {feat} → {cn[p].upper()} ({pr[p]*100:.1f}%) {ok}")
        pause(0.2)
    print(f"\n  ✓ Classification Demo Complete | Accuracy:{acc*100:.2f}% F1:{f1*100:.2f}%")

# ── DEMO P3 ──────────────────────────────────────────────────
def demo_recommendation():
    sep("DEMO: PROJECT 3 — AI RECOMMENDATION LOGIC")
    from recommender import load_knowledge_base, build_tfidf_matrix, build_user_vector, score_all_roles, get_top_n, display_recommendations
    df = load_knowledge_base()
    mat, vec = build_tfidf_matrix(df)
    for profile in [{"name":"ML Enthusiast","skills":["python","machine_learning","tensorflow","pandas","deep_learning"]},
                    {"name":"Web Developer","skills":["javascript","react","nodejs","html","docker"]},
                    {"name":"Cloud+DevOps",  "skills":["aws","docker","kubernetes","linux","ci_cd"]}]:
        sep(f"  Profile: {profile['name']}")
        uv = build_user_vector(profile["skills"], vec)
        if uv is not None:
            display_recommendations(get_top_n(score_all_roles(uv, mat, df), 3), profile["skills"])
        pause(0.4)
    print("  ✓ Recommendation Demo Complete")

# ── DEMO P4 ──────────────────────────────────────────────────
def demo_recognition():
    sep("DEMO: PROJECT 4 — IMAGE/TEXT RECOGNITION")
    print("  Running OCR Pipeline (Path 1) with synthetic test image...\n")
    from recognition import run_recognition
    run_recognition(path_choice=1)
    print("  ✓ Recognition Demo Complete")


# ── REQUESTED OUTPUT RUNNER ──────────────────────────────────
def demo_chatbot_requested():
    sep("REQUESTED OUTPUT — PROJECT 1 CHATBOT")
    from chatbot_v2 import get_response, sanitize

    conversation_history = []
    groq_client = None  # deterministic L4 fallback for terminal output
    demo_inputs = [
        "hello",
        "what is machine learning",
        "why does model training sometimes fail even when the data looks clean?",
    ]

    print("  3-turn conversation with visible source labels:\n")
    for i, raw_input in enumerate(demo_inputs, start=1):
        response, source = get_response(sanitize(raw_input), groq_client, conversation_history, demo_mode=True)
        print(f"  Turn {i}")
        print(f"    You  : {raw_input}")
        print(f"    DBot : {response}")
        print(f"            {source}\n")
        conversation_history.append({"role": "user", "content": raw_input})
        conversation_history.append({"role": "assistant", "content": response})


def demo_classification_requested():
    sep("REQUESTED OUTPUT — PROJECT 2 CLASSIFICATION")
    from classifier import load_data, split_data, scale_features, train_model, evaluate_model

    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_sc, y_train, k=5)
    evaluate_model(model, X_test_sc, y_test, class_names)


def demo_recommendation_requested():
    sep("REQUESTED OUTPUT — PROJECT 3 RECOMMENDATION")
    from recommender import (
        load_knowledge_base,
        build_tfidf_matrix,
        build_user_vector,
        score_all_roles,
        get_top_n,
        display_recommendations,
    )

    sample_skills = ["python", "machine_learning", "tensorflow", "pandas", "sql"]
    print(f"  Sample skill input: {sample_skills}\n")
    df = load_knowledge_base()
    tfidf_matrix, vectorizer = build_tfidf_matrix(df)
    user_vector = build_user_vector(sample_skills, vectorizer)

    if user_vector is None:
        print("  No matching skills were found in the recommendation vocabulary.")
        return

    scored_df = score_all_roles(user_vector, tfidf_matrix, df)
    top_results = get_top_n(scored_df, n=3)
    display_recommendations(top_results, sample_skills)


def demo_recognition_requested():
    sep("REQUESTED OUTPUT — PROJECT 4 RECOGNITION")
    from recognition import run_recognition_demo

    result = run_recognition_demo()
    if not result:
        print("  Recognition demo could not run because a dependency is missing.")
        return

    print("\n  Extracted OCR text:")
    for line in result["ocr_text"]:
        print(f"    - {line}")

    print("\n  Saved artifacts:")
    print(f"    - OCR binary image      : {result['ocr_binary_path']}")
    print(f"    - OCR text panel image  : {result['ocr_panel_path']}")
    print(f"    - Detection image       : {result['detection_output_path']}")

    print("\n  Detection summary:")
    for item in result["detections"]:
        print(
            f"    - {item['label']} | {item['confidence']*100:.1f}% | "
            f"BBox X={item['x']}, Y={item['y']}, W={item['w']}, H={item['h']}"
        )


def run_requested_output_demo():
    print("\n" + "█"*60)
    print("  DECODELABS AI SUITE — REQUESTED OUTPUT DEMO")
    print("  Chatbot + Classification + Recommendation + Recognition")
    print("█"*60)

    demo_chatbot_requested()
    pause(0.8)
    demo_classification_requested()
    pause(0.8)
    demo_recommendation_requested()
    pause(0.8)
    demo_recognition_requested()

    print("\n" + "█"*60)
    print("  REQUESTED OUTPUT DEMO COMPLETE ✓")
    print("█"*60 + "\n")

# ── MASTER RUNNER ─────────────────────────────────────────────
def run_demo():
    print("\n"+"█"*60)
    print("  DECODELABS AI SUITE — AUTOMATED DEMO")
    print("  All 4 Projects | No Typing Needed | Batch 2026")
    print("█"*60)
    print("""
  [1] Project 1 — Chatbot V1 (Pure Rule-Based)
  [2] Project 1 — Chatbot V2 (Hybrid + Groq layers)
  [3] Project 2 — KNN Classification (Iris)
  [4] Project 3 — TF-IDF Recommendation Engine
  [5] Project 4 — Image/Text Recognition (OCR + Detection)
    """)
    pause(1)
    demo_chatbot_v1();    pause(1)
    demo_chatbot_v2();    pause(1)
    demo_classification();pause(1)
    demo_recommendation();pause(1)
    demo_recognition()
    print("\n"+"█"*60)
    print("  ALL DEMOS COMPLETE ✓ | DecodeLabs AI Suite | Batch 2026")
    print("  Contact: decodelabs.tech@gmail.com | www.decodelabs.tech")
    print("█"*60+"\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"requested", "report", "output"}:
        run_requested_output_demo()
    else:
        run_demo()
