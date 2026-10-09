#!/usr/bin/env python3
"""
Simple test script for workflow system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test basic imports"""
    try:
        print("Testing workflow_models import...")
        from core.workflow_models import OutputFormat, PersonaType, WorkflowStrategy

        print("✅ workflow_models imported successfully")

        print("Testing workflow_generator import...")
        from core.workflow_generator import PRDParser

        print("✅ workflow_generator imported successfully")

        print("Testing workflow_formatter import...")
        from core.workflow_formatter import WorkflowFormatter

        print("✅ workflow_formatter imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality"""
    try:
        from core.workflow_generator import PRDParser
        from core.workflow_models import PRDStructure, Requirement

        print("Testing PRD parser...")
        parser = PRDParser()

        simple_text = "实现用户登录功能。支持邮箱密码登录。"
        prd = parser.parse_text_description(simple_text)

        print(f"✅ Parsed {len(prd.requirements)} requirements")
        print(f"   Title: {prd.title}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Simple Workflow System Test")
    print("=" * 50)

    success = True

    # Test imports
    if not test_imports():
        success = False

    print()

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    print()
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
