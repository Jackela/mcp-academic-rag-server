#!/usr/bin/env python3
"""
Test just the enum imports
"""

print("Testing enum imports...")

try:
    from enum import Enum
    print("✅ Built-in enum works")
    
    # Test our enums one by one
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    print("Testing WorkflowStrategy...")
    from core.workflow_models import WorkflowStrategy
    print(f"✅ WorkflowStrategy: {[s.value for s in WorkflowStrategy]}")
    
    print("Testing OutputFormat...")
    from core.workflow_models import OutputFormat
    print(f"✅ OutputFormat: {[f.value for f in OutputFormat]}")
    
    print("Testing PersonaType...")
    from core.workflow_models import PersonaType
    print(f"✅ PersonaType: {[p.value for p in PersonaType]}")
    
    print("All enum tests passed!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()