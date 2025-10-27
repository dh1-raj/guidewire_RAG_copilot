#!/bin/bash
# Setup script for RAG CLI shortcuts

PROJECT_DIR="/Users/dhiraj/Github copilot builds/Project_1"

echo "🚀 Setting up RAG CLI shortcuts..."
echo ""

# 1. Create alias in .zshrc
ALIAS_LINE="alias rag='cd \"$PROJECT_DIR\" && source venv/bin/activate && python rag_cli.py'"

if grep -q "alias rag=" ~/.zshrc; then
    echo "✓ 'rag' alias already exists in ~/.zshrc"
else
    echo "$ALIAS_LINE" >> ~/.zshrc
    echo "✓ Added 'rag' alias to ~/.zshrc"
fi

# 2. Create a standalone executable script
cat > ~/bin/rag << EOF
#!/bin/bash
cd "$PROJECT_DIR"
source venv/bin/activate
python rag_cli.py "\$@"
EOF

# Make it executable
chmod +x ~/bin/rag 2>/dev/null || {
    mkdir -p ~/bin
    chmod +x ~/bin/rag
}

echo "✓ Created ~/bin/rag executable"
echo ""

# 3. Add ~/bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
    echo "✓ Added ~/bin to PATH in ~/.zshrc"
else
    echo "✓ ~/bin already in PATH"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete! You now have multiple ways to query your knowledge base:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 Option 1: Interactive Mode (recommended)"
echo "   Just type: rag"
echo "   Then ask questions interactively"
echo ""
echo "📌 Option 2: Quick Query"
echo "   rag \"your question here\""
echo "   Example: rag \"event driven architecture best practices\""
echo ""
echo "📌 Option 3: Original Command (still works)"
echo "   python search_rag.py \"your question\""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  IMPORTANT: Run this command to activate the alias in your current shell:"
echo "   source ~/.zshrc"
echo ""
echo "Or simply open a new terminal window."
echo ""
EOF

chmod +x setup_rag_cli.sh
echo "Setup script created!"
echo ""
echo "🎯 Next steps:"
echo "   1. Run: source ~/.zshrc"
echo "   2. Type: rag"
echo "   3. Start asking questions!"
