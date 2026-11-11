import sys
from volatility3.cli import main

if __name__ == '__main__':
    # Set sys.argv[0] to not confuse argparse
    sys.argv[0] = 'volatility3'
    main()
