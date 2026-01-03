#!/bin/bash
# Diagnostic script for web search issues

echo "=========================================="
echo "Web Search Diagnostic Tool"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "1. Checking virtual environment..."
if [ -d ".venv" ]; then
    echo "   ✓ .venv exists"
else
    echo "   ✗ .venv NOT FOUND"
    exit 1
fi

echo ""
echo "2. Activating virtual environment..."
source .venv/bin/activate

echo "   ✓ Virtual environment activated"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

echo ""
echo "3. Checking ddgs package..."
python -c "import ddgs; print('   ✓ ddgs package found'); print(f'   Version: {ddgs.__version__ if hasattr(ddgs, \"__version__\") else \"unknown\"}')" 2>&1 || echo "   ✗ ddgs NOT installed"

echo ""
echo "4. Testing DDGS import..."
python -c "from ddgs import DDGS; print('   ✓ DDGS class imported successfully')" 2>&1 || echo "   ✗ DDGS import failed"

echo ""
echo "5. Testing actual web search..."
python -c "
from ddgs import DDGS
try:
    with DDGS() as ddgs:
        results = list(ddgs.text('test', max_results=1))
        print(f'   ✓ Web search works! Got {len(results)} result(s)')
        if results:
            print(f'   Result: {results[0][\"title\"][:50]}...')
except Exception as e:
    print(f'   ✗ Web search failed: {e}')
" 2>&1

echo ""
echo "6. Testing via tool executor..."
python -c "
from kautilya.tool_executor import ToolExecutor
executor = ToolExecutor(config_dir='.kautilya')
result = executor.execute('web_search', {'query': 'test', 'max_results': 1})
if result.get('success'):
    print(f'   ✓ Tool executor works! Found {result.get(\"result_count\")} results')
else:
    print(f'   ✗ Tool executor failed: {result.get(\"error\")}')
" 2>&1

echo ""
echo "7. Checking requirements.txt..."
if grep -q "ddgs" requirements.txt; then
    echo "   ✓ ddgs is in requirements.txt"
else
    echo "   ✗ ddgs NOT in requirements.txt"
    echo "   Adding ddgs to requirements.txt..."
    echo "ddgs>=9.0.0" >> requirements.txt
    echo "   ✓ Added ddgs to requirements.txt"
fi

echo ""
echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "If all checks passed but web search still doesn't work,"
echo "try running:"
echo "  source .venv/bin/activate"
echo "  uv pip install --force-reinstall ddgs"
echo "  ./run.sh"
