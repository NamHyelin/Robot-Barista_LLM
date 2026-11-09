import pyttsx3
import requests
import time

from image_processing import ImageProcessing, CameraInterface
from control import RobotServer
# from task_planning import Scoring

server = RobotServer(host="192.168.137.50", port=20002)
url_save = 'saved.jpg'
ci = CameraInterface(url=url_save)
ip = ImageProcessing('saved.jpg')

# scoring = Scoring("./gpt2_recipe_generation")

global_target_x = 640
global_target_y = 408
global_margin = 15
SERVER_HOST = "165.132.40.52"     # your Flask host
SERVER_PORT = 5002
ENDPOINT    = f"http://{SERVER_HOST}:{SERVER_PORT}/hybrid_inference_partial"

def main():
    prompt_list = [
        'Place cup',
        'Add ice',
        #'Drizzle caramel',
        'Drizzle cocoa',
        'Pour milk',
        #'Pour espresso',
        'Garnish caramel',
        'Serve Beverage',
        'Done'
    ]
    # 통신 시작
    server.start()
    # 컵의 현재 위치 저장
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    '''
    # 리스트에 저장된 재료 동작을 순차적으로 처리
    prompt = "<recipe_generation> Make me a caramel macchiato <recipe_generation> 1."
    # count = 2
    payload = {
            "prompt": prompt,
            "max_new_tokens": 60,
            "uncertainty_threshold": 0.15,
            "verbose": True
        }
    
    try:
        resp = requests.post(ENDPOINT, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        exit(1)
    print(result)
    '''
    '''
    prompt = prompt + result["partial_text"] + str(count)
    print(prompt)

    if "done" in result["partial_text"].lower():
    '''
    while prompt_list:
        print("next action")
        ''' 
        while count < 10:
        payload = {
            "prompt": prompt,
            "max_new_tokens": 50,
            "uncertainty_threshold": 0.05,
            "verbose": True
        }

        try:
            resp = requests.post(ENDPOINT, json=payload, timeout=15)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
            exit(1)
        if result["partial_text"].startswith(". "):
            motion = result["partial_text"][2:].strip()
        else: 
            motion = result["partial_text"].strip()
        
        prompt = prompt + result["partial_text"] + str(count)
        print(prompt)

        if "done" in result["partial_text"].lower():
            break

        count = count + 1
        parts = motion.split(" ", 1)
        '''

        task = prompt_list.pop(0)
        if task.lower() == "done":
            break
        # "동작 재료" 형태로 split (예: "Place Cup" -> ['Place', 'Cup'])
        parts = task.split(" ", 1)
        action = parts[0]

        ingredient = parts[1] if len(parts) > 1 else None

        current_action = action.lower()  # 동작을 소문자로 변환하여 전달

        print(current_action)
        print(ingredient)

        if ingredient is None:
            server.send(current_action)
            server.rbt_wait()
            continue
        # 미리 컵들이 있는 위치로 이동
        if current_action == "place":
            server.send("cup_place")
            server.rbt_wait()

        elif current_action == "serve":
            server.send("to_beverage")
            server.rbt_wait()
        else:
            server.send("prepare")
            server.rbt_wait()
        # 재료가 없으면 2번까지 반복
        retry = 2
        found = False
        for attempt in range(retry + 1):
            ci.capture_image()
            found, cx, cy = ip.find_ingredient_cup(ingredient)
            if found:
                print(f'[FOUND on try {attempt+1}/{retry+1}] {ingredient} at {cx}, {cy}')
                break
            else:
                print(f"[FOUND on try {attempt+1}/{retry+1}] '{ingredient}' not found, retry...")
                time.sleep(0.1)

        if not found:
            print(f"[SKIP] '{ingredient}' not found after {retry+1} attempts.")

            continue

        while True:
            ci.capture_image()
            # 재료 탐색 함수 호출 (모든 재료에 대해 동일하게 작동)
            found, cx, cy = ip.find_ingredient_cup(ingredient)
            if found:
                alignment = move(cx,cy)
                if alignment:
                    print("Aligned!")
                    break
            else:
                print("Not Alligned...")
                continue

        if found and alignment:
        # engine.say(f"{action} {ingredient}")
        #engine.runAndWait()
            server.send(current_action)
            print("action :",current_action, ingredient)
            server.rbt_wait()
            if current_action == "place":
                while True:
                    ci.capture_image()
                # 컵을 내려놓고, 위치를 저장하기 위해 정렬
                    found, cx, cy = ip.detect_rim_hough()
                    if found:
                        alignment = move(cx,cy)
                        if alignment:
                            print("Aligned!")
                            break
                    else:
                        continue
                server.send("save")
                print("save cup location")
                server.rbt_wait()

            elif current_action == "serve":
                user_input = input("Ready to receive? type 'ok' ")
                if user_input == 'ok':
                    server.send("ok")
                    server.rbt_wait()

    print("complete!")
    engine.say("Complete!")
    engine.runAndWait()

def move(cx: int, cy: int, target_x=global_target_x, target_y=global_target_y, margin=global_margin):
    alignment = False
    if ((target_x - margin) <= cx <= (target_x + margin)) and ((target_y - margin) <= cy <= (target_y + margin)):
        alignment = True
    else:
        server.move_delta(cx=target_x-cx, cy=target_y-cy)

    return alignment

if __name__ == "__main__":
    main()