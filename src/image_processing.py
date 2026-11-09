import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np

# Access camera and save the image in jpg format
class CameraInterface:    
    def __init__(self, url='saved.jpg'):
        self.cam1 = 0
        self.cap = cv2.VideoCapture(self.cam1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        self.url = url

    def capture_image(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        ret, img = self.cap.read()
        img = cv2.resize(img, (1280, 916))
        cv2.imwrite(self.url, img)

    def return_image(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2748)
        ret, img = self.cap.read()
        img = cv2.resize(img, (1280, 916))
        return img

# Returns image coordinates of target objects from a saved image
class ImageProcessing:
    def __init__(self, url='saved.jpg'):
        self.lower_red1 = np.array([0, 120, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 150])
        self.upper_red2 = np.array([180, 255, 255])
        self.upper_white = np.array([255, 255, 255])
        self.lower_white = np.array([0, 0, 200])

        self.url = url
        self.ci = CameraInterface()
    
    def find_ingredient_cup(self, target_word="Water", max_attempts=2):
        model = ocr_predictor(pretrained=True)

        for attempt in range(max_attempts):
            # 1) 이미지 캡처 + OCR
            self.ci.capture_image()
            doc   = DocumentFile.from_images(self.url)
            result= model(doc).export()
            target_x = None

            for page in result["pages"]:
                pw = page["dimensions"][1]
                for block in page["blocks"]:
                    for line in block["lines"]:
                        for word in line["words"]:
                            if word["value"].lower() == target_word.lower():
                                (x1, _), (x2, _) = word["geometry"]
                                target_x = ((x1 + x2) / 2) * pw
                                break
                        if target_x is not None:
                            break
                    if target_x is not None:
                        break
                if target_x is not None:
                    break

            # 2) 단어를 못 찾았으면 다음 시도로
            if target_x is None:
                continue

            # 3) 찾았으면 바로 로그 출력 후 원 검출
            print(f"Found word '{target_word}' at x={target_x:.0f}")
            img   = cv2.imread(self.url)
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (9,9), 2)
            edges = cv2.Canny(blur, 50, 150)
            circles = cv2.HoughCircles(
                edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                param1=150, param2=30, minRadius=100, maxRadius=160
            )

            if circles is not None:
                c = min(circles[0].astype(np.float32),
                        key=lambda c: abs(c[0] - target_x))
                cx, cy, r = map(int, c)
                out = cv2.imread(self.url)
                cv2.circle(out, (cx, cy), r, (0,255,0), 2)
                cv2.circle(out, (cx, cy), 2, (0,255,0), 3)
                cv2.imwrite(self.url, out)
                return True, cx, cy

            # 원 자체를 못 찾으면 False로 즉시 반환
            return False, None, None

        # max_attempts 내내 단어를 못 찾았으면
        return False, None, None


    def detect_rim_hough(self, img_path = 'saved.jpg'):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)
        # 1) Canny 에지 검출: 컵 테두리만 선명하게
        edges = cv2.Canny(blur, 50, 150)
        # 2) HoughCircles: 에지에서만 원 찾기
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=150,   # Canny 상단 임계치
            param2=30,       # 투표 임계치(낮으면 많은 원, 높으면 강한 원)
            minRadius=80,
            maxRadius=170
        )
        if circles is not None:
            c = np.uint16(np.around(circles))[0,0]
            image = cv2.imread('saved.jpg')
            cv2.circle(image, (c[0],c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(image, (c[0],c[1]), 2, (0, 255, 0), 3)
            cv2.imwrite('saved.jpg', image)
            print(f"Circle center at x={c[0]}, y={c[1]}")
            return True, int(c[0]), int(c[1])
        return False, None, None

if __name__ == "__main__":
    ip = ImageProcessing()
    ip.find_ingredient_cup("caramel")