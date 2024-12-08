from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import pandas as pd
import open_clip
# Initialize the Flask app
app = Flask(__name__)

# Load the model, tokenizer, and preprocess
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai', cache_dir='cache')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Load the database of image embeddings
df = pd.read_pickle('image_embeddings.pickle')

# Helper functions
def get_top_k_results(query_embedding, top_k=5):
    cosine_similarities = df['embedding'].apply(
        lambda x: F.cosine_similarity(query_embedding, torch.from_numpy(x), dim=0).item()
    )
    top_k_indices = cosine_similarities.nlargest(top_k).index
    results = [
        {
            "file_name": df.loc[idx, 'file_name'],
            "similarity": cosine_similarities[idx]
        }
        for idx in top_k_indices
    ]
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query_type = request.form.get("query_type")
    top_k = int(request.form.get("top_k", 5))

    # Handle different query types
    if query_type == "text":
        text_query = request.form.get("text_query")
        text = tokenizer([text_query])
        query_embedding = F.normalize(model.encode_text(text)).squeeze(0)

    elif query_type == "image":
        image = request.files["image_query"]
        image = preprocess(Image.open(image)).unsqueeze(0)
        query_embedding = F.normalize(model.encode_image(image)).squeeze(0)

    elif query_type == "combined":
        text_query = request.form.get("text_query")
        lam = float(request.form.get("weight", 0.5))

        # Image query
        image = request.files["image_query"]
        image = preprocess(Image.open(image)).unsqueeze(0)
        image_query = F.normalize(model.encode_image(image))

        # Text query
        text = tokenizer([text_query])
        text_query = F.normalize(model.encode_text(text))

        # Combine queries
        query_embedding = F.normalize(lam * text_query + (1.0 - lam) * image_query).squeeze(0)

    else:
        return jsonify({"error": "Invalid query type"}), 400

    # Get results
    results = get_top_k_results(query_embedding, top_k)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)