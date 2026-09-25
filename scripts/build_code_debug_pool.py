#!/usr/bin/env python3
"""SWE-bench Verified prompt preparation; see prepare_inputs.py --help."""
import sys
from prepare_inputs import main
if __name__=='__main__':
    if '--domain' not in sys.argv:sys.argv.extend(['--domain','code_debug'])
    if '--count' not in sys.argv:sys.argv.extend(['--count','100'])
    main()
