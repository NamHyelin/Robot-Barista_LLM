import os
import json
import random
import time

from datasets import Dataset
from google import genai
from google.genai.errors import ClientError


# Available actions: pour, grab, place, serve, done
class DataGeneration():
    def __init__(self, dataset_filename = "data_collection.json"):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.dataset_filename = dataset_filename
        if os.path.exists(dataset_filename):
            with open(dataset_filename, "r") as f:
                self.distillation_data = json.load(f)
        else:
            self.distillation_data = []

    def generate_caramel_macchiato(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an iced caramel macchiato order with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label =  "1. Place Cup 2. Drizzle Vanilla Syrup 3. Pour Milk 4. Add Ice 5. Pour Espresso 6. Pour Espresso 7. Drizzle Caramel Syrup 8. Serve Beverage 9. Done"
            else:
                prompt = "Produce a user request for an iced caramel macchiato order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Drizzle Vanilla Syrup 3. Pour Milk 4. Add Ice 5. Pour Espresso 6. Drizzle Caramel Syrup 7. Serve Beverage 8. Done"
        else: 
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for a caramel macchiato order with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Drizzle Vanilla Syrup 3. Pour Milk 4. Pour Espresso 5. Pour Espresso 6. Drizzle Caramel Syrup 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for a caramel macchiato order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Drizzle Vanilla Syrup 3. Pour Milk 4. Pour Espresso 5. Drizzle Caramel Syrup 6. Serve Beverage 7. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt])
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            # "response": ["grab cup", "place cup", "pour espresso", "pour milk", "pour caramel", "serve", "done"]
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
        
    def generate_americano(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an iced americano order with an extra shot of espresso in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Add Ice 4. Pour Espresso 5. Pour Espresso 6. Pour Espresso 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for an iced americano order in a cafe. Return nothing but the order."
                label =  "1. Place Cup 2. Pour Water 3. Add Ice 4. Pour Espresso 5. Pour Espresso 6. Serve Beverage 7. Done"
        else: 
            if extra_espresso: 
                prompt = "You are a customer at a cafe. Produce a user request for an americano order with an extra shot of espresso in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Pour Espresso 5. Pour Espresso 6. Serve Beverage 7. Done"
            else:
                prompt = "Produce a user request for an americano order in a cafe. Return nothing but the order."
                label = "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Pour Espresso 5. Serve Beverage 6. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt])
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
    
    def generate_cafe_mocha(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for an iced cafe mocha order "
                        "with an extra shot of espresso in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Drizzle Chocolate Syrup 5. Pour Milk 6. Add Ice 7. Garnish Cocoa powder 8. Serve Beverage 9. Done"
            else:
                prompt = "Produce a user request for an iced cafe mocha order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Drizzle Chocolate Syrup 4. Pour Milk 5. Add Ice 6. Garnish Cocoa powder 7. Serve Beverage 8. Done"
        else:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for a cafe mocha order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Chocolate 5. Pour Milk 6. Garnish Cocoa powder 7. Serve Beverage 8. Done"
            else:
                prompt = "Produce a user request for a cafe mocha order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Chocolate 4. Pour Milk 5. Garnish Cocoa powder 6. Serve Beverage 7. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)

    def generate_cafe_latte(self, extra_espresso=False, ice=False):
        if ice:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for an iced cafe latte order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Add Ice 6. Serve Beverage 7. Done"
            else:
                prompt = "Produce a user request for an iced cafe latte order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Milk 4. Add Ice 5. Serve Beverage 6. Done"
        else:
            if extra_espresso:
                prompt = ("You are a customer at a cafe. Produce a user request for a cafe latte order. Do not replace milk with other types like oat milk."
                        "with an extra shot of espresso in a cafe. Return nothing but the order.")
                label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Serve Beverage 6. Done"
            else:
                prompt = "Produce a user request for a cafe latte order in a cafe. Return nothing but the order. Do not replace milk with other types like oat milk."
                label = "1. Place Cup 2. Pour Espresso 3. Pour Milk 4. Serve Beverage 5. Done"
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)

        self.distillation_data.append({
            "prompt": sentence,
            "response": label
        })
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
    '''
    def generate_java_chip_frappuccino(self, extra_espresso=False, extra_java_chip=False):
        if extra_espresso and extra_java_chip:
            prompt = "You are a customer at a cafe. Produce a user request for a java chip frappuccino order with an extra shot of espresso and extra java chips. Return nothing but the order."
            label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Espresso 5. Add Java Chips 6. Add Java Chips 7. Pour Milk 8. Add Ice 9. Blend Beverage 10. Serve Beverage 11. Done"
        elif extra_espresso:
            prompt = "You are a customer at a cafe. Produce a user request for a java chip frappuccino order with an extra shot of espresso. Return nothing but the order."
            label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Espresso 5. Add Java Chips 6. Pour Milk 7. Add Ice 8. Blend Beverage 9. Serve Beverage 10. Done"
        elif extra_java_chip:
            prompt = "You are a customer at a cafe. Produce a user request for a java chip frappuccino order with extra java chips. Return nothing but the order."
            label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Add Java Chips 6. Add Java Chips 7. Add Ice 8. Blend Beverage 9. Serve Beverage 10. Done"
        else:
            prompt = "Produce a user request for a java chip frappuccino order in a cafe. Return nothing but the order."
            label = "1. Place Cup 2. Pour Espresso 3. Pour Espresso 4. Pour Milk 5. Add Java Chips 6. Add Ice 7. Blend Beverage 8. Serve Beverage 9. Done"
        try:
            response = self.client.models.generate_content(model="gemini-2.0-flash-thinking-exp-01-21", contents=[prompt])
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return
        words = response.text.split()
        sentence = " ".join(words)
        self.distillation_data.append({"prompt": sentence, "response": label})
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)

    def generate_matcha_frappuccino(self):
        prompt = "You are a customer at a cafe. Produce a user request for a matcha frappuccino order. Return nothing but the order."
        label = "1. Place Cup 2. Pour Milk 3. Add Matcha Powder 4. Add Sugar 5. Add Ice 6. Blend Beverage 7. Add Whipped Cream 8. Serve Beverage 9. Done"
        
        try:
            # Generate the content using the specified model.
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting...")
            time.sleep(10)
            return

        # Process the response to form a single sentence.
        words = response.text.split()
        sentence = " ".join(words)
        
        # Append the prompt and label to the distillation data.
        self.distillation_data.append({"prompt": sentence, "response": label})
        
        # Save the collected data to a JSON file.
        with open(self.dataset_filename, "w") as f:
            import json  # Ensure json module is imported if not available globally.
            json.dump(self.distillation_data, f, indent=4)

    def generate_blue_lemonade(self):
        prompt = "You are a customer at a cafe. Produce a user request for a blue lemonade order. Return nothing but the order."
        
        # Define the canonical order steps for Blue Lemonade.
        label = "1. Place Cup 2. Add Ice 3. Pour Lemon Juice 4. Pour Water 5. Pour Blue Syrup 6. Serve Beverage 7. Done"
        
        try:
            # Generate the content using the specified model.
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                contents=[prompt]
            )
        except ClientError as e:
            print("Waiting for the client to be ready...")
            time.sleep(10)
            return

        # Process the response text into a clean sentence.
        words = response.text.split()
        sentence = " ".join(words)
        
        # Append the prompt (generated order) and its corresponding label to the dataset.
        self.distillation_data.append({"prompt": sentence, "response": label})
        
        # Save the collected data to a JSON file.
        with open(self.dataset_filename, "w") as f:
            json.dump(self.distillation_data, f, indent=4)
    '''
    def generate(self):
        recipe_selection = random.randint(1, 10)
        ice = random.randint(1,10) > 7
        extra_espresso = random.randint(1,10) > 7
        
        if recipe_selection <= 2:
            self.generate_americano(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection <= 4:
            self.generate_cafe_latte(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection <= 7:
            self.generate_cafe_mocha(ice=ice, extra_espresso=extra_espresso)
        elif recipe_selection <= 10:
            self.generate_caramel_macchiato(ice=ice, extra_espresso=extra_espresso)
        else:
            pass

    def data_collection(self):
        for i in range(1, 1000):
            print(f"Processing: {i}...")
            self.generate()
            
dg = DataGeneration()
dg.data_collection()
