#!/usr/bin/env bash

echo "Generating documentation..."
doxygen Doxyfile && \
echo "Documentation available at http://localhost:8000" && \
python3 -m http.server --directory "docs/html" 8000
