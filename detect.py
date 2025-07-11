import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-8.3.9\runs\train\ContainerYOLO8\weights\last.pt')

    results = model.predict(
        source=r'C:\Users\admin\Desktop\damage02-768x576.jpg',
        imgsz=640,
        device='0',
        save=False,
        stream=True
    )

    for result in results:
        print(result)
