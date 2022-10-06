import sys
import ast

try:
    ast.parse(sys.argv[1])  # Try to parse the string.
except SyntaxError as err:
    print(err)