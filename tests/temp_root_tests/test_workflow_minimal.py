#!/usr/bin/env python3
"""
Minimal workflow system test to avoid circular dependency
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_models_only():
    """Test just the data models"""
    try:
        from core.workflow_models import (
            WorkflowStrategy, OutputFormat, PersonaType, Priority,
            Requirement, PRDStructure, Workflow
        )
        
        print("✅ Successfully imported workflow models")
        
        # Create a simple requirement
        req = Requirement(
            title="Test Requirement",
            description="A simple test requirement",
            priority=Priority.MEDIUM
        )
        
        print(f"✅ Created requirement: {req.title}")
        
        # Create a simple PRD structure
        prd = PRDStructure(
            title="Test PRD",
            overview="Test overview",
            requirements=[req],
            date_created=datetime.now()
        )
        
        print(f"✅ Created PRD: {prd.title} with {len(prd.requirements)} requirements")
        
        # Create a simple workflow without phases to avoid circular dependency
        workflow = Workflow(
            name="Test Workflow",
            description="A test workflow",
            strategy=WorkflowStrategy.SYSTEMATIC,
            requirements=[req],
            prd_structure=prd
        )
        
        print(f"✅ Created workflow: {workflow.name}")
        print(f"   Strategy: {workflow.strategy.value}")
        print(f"   Requirements: {len(workflow.requirements)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Models test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_formatter_only():
    """Test just the formatter"""
    try:
        from core.workflow_formatter import WorkflowFormatter
        from core.workflow_models import Workflow, WorkflowStrategy, OutputFormat
        
        print("✅ Successfully imported workflow formatter")
        
        # Create minimal workflow
        workflow = Workflow(
            name="Test Workflow",
            description="Test description",
            strategy=WorkflowStrategy.SYSTEMATIC
        )
        
        formatter = WorkflowFormatter()
        
        # Test roadmap format
        roadmap_output = formatter.format_workflow(workflow, OutputFormat.ROADMAP)
        print(f"✅ Generated roadmap format ({len(roadmap_output)} characters)")
        
        # Test tasks format  
        tasks_output = formatter.format_workflow(workflow, OutputFormat.TASKS)
        print(f"✅ Generated tasks format ({len(tasks_output)} characters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Formatter test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Minimal Workflow System Test")
    print("=" * 50)
    
    success = True
    
    # Test models
    print("Testing data models...")
    if not test_models_only():
        success = False
    
    print("\nTesting formatter...")
    if not test_formatter_only():
        success = False
    
    print()
    if success:
        print("🎉 Minimal tests passed!")
        print("The workflow system core components are working correctly.")
        print("The circular dependency issue needs to be fixed in the generator.")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)