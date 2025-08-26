#!/usr/bin/env python3
"""
DiaModel의 실제 구조를 확인하는 스크립트
"""

import torch
from transformers import AutoModel

def print_model_structure():
    """DiaModel의 구조를 출력합니다."""
    print("Loading DiaModel...")
    model = AutoModel.from_pretrained("nari-labs/Dia-1.6B-0626")
    
    print("\n=== Model Structure ===")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    print("\n=== Top-level modules ===")
    for name, module in model.named_children():
        print(f"{name}: {type(module)}")
    
    print("\n=== All modules (first 20) ===")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 20:
            print("...")
            break
        print(f"{name}: {type(module)}")
    
    print("\n=== Linear modules (potential LoRA targets) ===")
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append(name)
            print(f"{name}: {module}")
    
    print(f"\nFound {len(linear_modules)} Linear modules")
    
    # 패턴 매칭 확인
    print("\n=== Checking problematic pattern ===")
    problematic_pattern = "16.self_attention.k_proj"
    print(f"Looking for pattern: {problematic_pattern}")
    
    found = False
    for name, module in model.named_modules():
        if problematic_pattern in name:
            print(f"Found: {name}")
            found = True
    
    if not found:
        print("Pattern not found in model structure")
        
        # 대신 self_attention이 포함된 모듈들 찾기
        print("\n=== Modules containing 'self_attention' ===")
        for name, module in model.named_modules():
            if 'self_attention' in name:
                print(f"{name}: {type(module)}")

if __name__ == "__main__":
    print_model_structure()
