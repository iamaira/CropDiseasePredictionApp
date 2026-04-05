import torch
from acfg.appconfig import CLF_MODEL, ServiceConfig

# Print model info
print("=== Model Information ===")
print(f"Model type: {type(CLF_MODEL)}")
print(f"In eval mode: {not CLF_MODEL.training}")
params = sum(p.numel() for p in CLF_MODEL.parameters())
print(f"Total parameters: {params}")

# Check model state
print("\n=== Layer Info (first 3 parameters) ===")
for i, (name, param) in enumerate(list(CLF_MODEL.named_parameters())[:3]):
    print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}, mean={param.mean():.6f}")

# Test with multiple random inputs to see if outputs vary
print("\n=== Testing with multiple random inputs ===")
for test_num in range(3):
    random_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = CLF_MODEL(random_img)
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, 5)
    
    print(f"\nTest {test_num + 1}:")
    print(f"  Top prediction: {ServiceConfig.ID2LABEL[top5.indices[0].item()]} ({top5.values[0].item():.4f})")
    print(f"  Top 5:")
    for idx, (prob, class_idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"    {idx+1}. {ServiceConfig.ID2LABEL[class_idx.item()]} ({prob.item():.4f})")
