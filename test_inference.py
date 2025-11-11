from amd_ai_optimization.rocm_inference import OptimizedInferenceEngine
import numpy as np

engine = OptimizedInferenceEngine()

# engine.load_model("model.pth", model_type="pytorch")

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
# output = engine.infer(input_data)

print("Inference engine initialized!")
print(f"Backend: {engine.backend.backend_type.value}")