import time
import numpy as np
from openvino.runtime import Core

ie = Core()
print("Dispositivos:", ie.available_devices)

model = ie.read_model("face-detection-retail-0004.xml")
compiled = ie.compile_model(model, "MYRIAD")
infer = compiled.create_infer_request()

# input dummy 1x3x300x300 (NCHW, FP32 — OpenVINO lo convierte a FP16 internamente)
x = np.random.rand(1, 3, 300, 300).astype(np.float32)

# warmup (la primera inferencia siempre es más lenta)
for _ in range(3):
    infer.infer({0: x})

# benchmark
N = 50
t0 = time.time()
for _ in range(N):
    infer.infer({0: x})
dt = time.time() - t0

print(f"MYRIAD: {N/dt:.1f} FPS ({dt*1000/N:.1f} ms/inferencia)")
