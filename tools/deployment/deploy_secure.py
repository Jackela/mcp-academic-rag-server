#!/usr/bin/env python3
"""
MCP Academic RAG Server - Setup and Deployment Script
Follows MCP 2024 best practices for secure deployment
"""

import getpass
import os
import subprocess
import sys
from pathlib import Path


def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key or not api_key.startswith("sk-") or len(api_key) < 20:
        return False
    return True


def setup_environment():
    """Setup environment with secure API key handling"""
    print("🔐 MCP Academic RAG Server - Setup")
    print("=" * 50)

    # Check if API key exists in environment
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key or not validate_api_key(api_key):
        print("❌ OpenAI API key not found or invalid in environment.")
        print("Please enter your OpenAI API key (input will be hidden):")

        try:
            api_key = getpass.getpass("API Key: ")
            if not validate_api_key(api_key):
                print("❌ Invalid API key format. Must start with 'sk-'")
                sys.exit(1)

            # Ask if user wants to save to .env file
            save_env = input("💾 Save to .env file for future use? (y/N): ").lower()
            if save_env == "y":
                env_content = f"""# MCP Academic RAG Server Environment Variables
OPENAI_API_KEY={api_key}
DATA_PATH=./data
LOG_LEVEL=INFO
# OCR_LANGUAGE=eng
# MCP_PORT=8000
"""
                with open(".env", "w") as f:
                    f.write(env_content)
                print("✅ Environment variables saved to .env file")
                print("💡 You can now use: source .env && uvx run mcp-academic-rag-server")
            else:
                print("✅ API key validated (session only)")
                print("💡 Remember to set OPENAI_API_KEY in your environment")

        except KeyboardInterrupt:
            print("\n❌ Setup cancelled")
            sys.exit(1)
    else:
        print("✅ OpenAI API key found and validated")

    # Setup data directory
    data_path = os.environ.get("DATA_PATH", "./data")
    Path(data_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Data directory ready: {data_path}")

    return api_key


def install_with_uvx():
    """Install package using uvx"""
    print("\n📦 Installing with uvx...")

    try:
        # Install package from current directory
        result = subprocess.run(["uvx", "install", "."], check=True, capture_output=True, text=True)
        print("✅ Package installed successfully")

        # Test installation
        result = subprocess.run(
            ["uvx", "run", "mcp-academic-rag-server", "--validate-only"],
            capture_output=True,
            text=True,
            env={**os.environ, "OPENAI_API_KEY": "sk-test123456789012345678"},
        )

        if "Environment validation completed" in result.stderr:
            print("✅ Installation test passed")
        else:
            print("⚠️  Installation test completed with warnings")

    except subprocess.CalledProcessError as e:
        print(f"❌ uvx installation failed: {e}")
        print("💡 Make sure uvx is installed: pip install uvx")
        sys.exit(1)


def show_configuration_guide(api_key: str):
    """Show configuration guide for Claude Desktop"""
    print("\n📋 Claude Desktop Configuration Guide")
    print("=" * 50)

    # Show config file location
    config_locations = {
        "macOS": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "Windows": "%APPDATA%\\Claude\\claude_desktop_config.json",
        "Linux": "~/.config/claude/claude_desktop_config.json",
    }

    print("📁 Configuration file locations:")
    for os_name, location in config_locations.items():
        print(f"  {os_name}: {location}")

    print("\n📝 Add this configuration:")

    config = f"""{{
  "mcpServers": {{
    "academic-rag": {{
      "command": "uvx",
      "args": ["run", "mcp-academic-rag-server"],
      "env": {{
        "OPENAI_API_KEY": "{api_key[:8]}...",
        "DATA_PATH": "./data",
        "LOG_LEVEL": "INFO"
      }}
    }}
  }}
}}"""
    print(config)

    print(f"\n⚠️  Replace '{api_key[:8]}...' with your full API key")
    print("🔐 Store your full API key securely in the configuration")

    # Show alternative methods
    print("\n📋 Alternative Configuration Methods:")
    print()
    print("1️⃣  Using .env file (if created):")
    print("   Run: source .env && uvx run mcp-academic-rag-server")
    print()
    print("2️⃣  Using environment variables:")
    print("   export OPENAI_API_KEY=sk-xxx...")
    print("   uvx run mcp-academic-rag-server")
    print()
    print("3️⃣  Direct command (for testing):")
    print("   OPENAI_API_KEY=sk-xxx... uvx run mcp-academic-rag-server --validate-only")


def show_usage_examples():
    """Show usage examples and testing"""
    print("\n🧪 Testing Your Installation")
    print("=" * 30)
    print()
    print("Test environment validation:")
    print("  uvx run mcp-academic-rag-server --validate-only")
    print()
    print("Run with custom settings:")
    print("  uvx run mcp-academic-rag-server --data-path /custom/path --log-level DEBUG")
    print()
    print("📖 More examples in README.md")


def main():
    """Main setup and deployment function"""
    try:
        print("🚀 MCP Academic RAG Server - Deployment Script")
        print("Following MCP 2024 best practices")
        print()

        # Setup environment
        api_key = setup_environment()

        # Install with uvx
        install_with_uvx()

        # Show configuration guide
        show_configuration_guide(api_key)

        # Show usage examples
        show_usage_examples()

        print("\n🎉 Setup completed successfully!")
        print("📋 Follow the configuration guide above to integrate with Claude Desktop")
        print("🔐 Remember: MCP servers require environment variables to be set before starting")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
