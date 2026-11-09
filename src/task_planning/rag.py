import json
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGPromptGenerator:
    def __init__(self, recipe_file: str, embedder_model: str = "all-MiniLM-L6-v2"):
        self.recipe_file = recipe_file
        self.embedder = SentenceTransformer(embedder_model)
        self.library = self._load_recipes(recipe_file)

    def _load_recipes(self, recipe_file: str):
        with open(recipe_file, "r", encoding="utf-8") as f:
            recipe_data = json.load(f)
        library = []
        for entry in recipe_data:
            beverage_name = entry.get("prompt", "")
            recipe_steps = entry.get("response", "")
            # Combine beverage name and recipe text for a more robust embedding
            combined_text = f"{beverage_name}. {recipe_steps}"
            embedding = self.embed_text(combined_text)
            library.append({
                "beverage": beverage_name,
                "recipe": recipe_steps,
                "embedding": embedding
            })
        return library

    def embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def retrieve_best_recipe(self, user_query: str, top_k: int = 1):
        query_emb = self.embed_text(user_query)
        scores = []
        for item in self.library:
            item_emb = item["embedding"]
            similarity = np.dot(query_emb, item_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(item_emb))
            scores.append((similarity, item))
        
        # Sort recipes by descending similarity
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = [s[1] for s in scores[:top_k]]
        # print(top_matches)
        return top_matches

    def build_prompt(self, user_query: str, recipe_entry: dict) -> str:
        beverage = recipe_entry["beverage"]
        original_recipe = recipe_entry["recipe"]
        
        # print("original " + original_recipe)

        prompt = f"""
        You are a coffee robot task planner. Your task is to generate a beverage preparation plan strictly following the exact format shown below in the original recipe.
        
        Important Instructions:
        - Use only these action verbs: "place", "pour", "serve", "drizzle", "garnish", "add", and "done".
        - Do not include any extra commentary or steps.
        - Maintain the exact numbering and structure: each line should start with a number, a period, and a space, followed by the action.
        - If modifications are needed (e.g., an extra shot of espresso), adjust by repeating the relevant step exactly within the sequence.
        - Refer to the original recipe steps for the recipe you need to create.
        
        Original Recipe Steps: {original_recipe}
        
        User Request: "{user_query}"
        
        Now, generate the final plan exactly following the format above.
    """
        return prompt.strip()

    def generate_rag_prompt(self, user_query: str) -> str:
        best_recipe = self.retrieve_best_recipe(user_query, top_k=1)[0]
        prompt = self.build_prompt(user_query, best_recipe)
        return prompt


if __name__ == "__main__":
    generator = RAGPromptGenerator(recipe_file="label.json")
    sample_query = "I want a Thai Tea Latte with an extra shot of espresso"
    final_prompt = generator.generate_rag_prompt(sample_query)
    print("=== Generated RAG Prompt ===")
    print(final_prompt)
