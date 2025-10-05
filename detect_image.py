from ultralytics import YOLO

model = YOLO("yolo12n.pt")

# image_p ath = "pasto_vacas.png"
image_path = "vacas.webp"

results = model(image_path)

detections = results[0]

# detections.show()

detections.save("result_bouding_box2.jpg")

print("Imagem salva como 'result_bouding_box.jpg'")
