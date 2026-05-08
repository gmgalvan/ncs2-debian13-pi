"""
Detector de caras sobre una imagen, usando NCS2.
Uso:
    python detect_image.py <ruta_local_o_url> [--out salida.jpg] [--threshold 0.5]
"""
import argparse
import sys
import urllib.request

import cv2
import numpy as np
from openvino.runtime import Core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Ruta a imagen local o URL http(s)://...")
    ap.add_argument("--out", default="output.jpg", help="Archivo de salida")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Confianza mínima (0-1)")
    ap.add_argument("--model", default="face-detection-retail-0004.xml")
    args = ap.parse_args()

    # 1. Conseguir la imagen (descargar si es URL)
    if args.input.startswith(("http://", "https://")):
        local = "input_downloaded.jpg"
        print(f"Descargando {args.input}...")
        req = urllib.request.Request(
            args.input,
            headers={"User-Agent": "Mozilla/5.0 (NCS2-lab)"}
        )
        with urllib.request.urlopen(req) as resp, open(local, "wb") as f:
            f.write(resp.read())
        img_path = local
    else:
        img_path = args.input

    image = cv2.imread(img_path)
    if image is None:
        sys.exit(f"No se pudo cargar {img_path}")
    h, w = image.shape[:2]
    print(f"Imagen: {w}x{h}")

    # 2. Cargar y compilar el modelo en MYRIAD
    ie = Core()
    print(f"Compilando {args.model} en MYRIAD...")
    model = ie.read_model(args.model)
    compiled = ie.compile_model(model, "MYRIAD")

    # 3. Preprocesar: resize a 300x300, layout HWC -> NCHW
    blob = cv2.resize(image, (300, 300))
    blob = blob.transpose((2, 0, 1))                   # HWC -> CHW
    blob = np.expand_dims(blob, 0).astype(np.float32)  # -> NCHW

    # 4. Inferir
    infer = compiled.create_infer_request()
    result = infer.infer({0: blob})

    # 5. Parsear la salida
    # Shape: [1, 1, 200, 7]
    # Cada fila: [image_id, label, conf, xmin, ymin, xmax, ymax] (normalizadas 0-1)
    detections = next(iter(result.values()))[0][0]
    print("detections: ", detections)
    print("type detections: ", type(detections))
    n_faces = 0
    for det in detections:
        conf = float(det[2])
        if conf < args.threshold:
            continue
        n_faces += 1
        xmin = int(det[3] * w)
        ymin = int(det[4] * h)
        xmax = int(det[5] * w)
        ymax = int(det[6] * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (xmin, max(ymin - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.out, image)
    print(f"{n_faces} cara(s) detectada(s) (umbral={args.threshold}). "
          f"Salida: {args.out}")


if __name__ == "__main__":
    main()