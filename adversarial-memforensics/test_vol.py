import sys
import subprocess
import os

dump = os.path.abspath(r'data\malvol25\tmp\Cerber-NoVirus.mem')
print('Running Volatility on', dump)

script = f'''
import sys
from volatility3.cli import main
sys.argv = [
    "vol.py",
    "-f", r"{dump}",
    "-r", "json",
    "windows.pslist.PsList"
]
main()
'''

with open('inner_vol.py', 'w') as f:
    f.write(script)

res = subprocess.run([sys.executable, 'inner_vol.py'], capture_output=True, text=True, timeout=120)
print('RC:', res.returncode)
print('STDOUT[:1000]:', res.stdout[:1000])
print('STDERR[:1000]:', res.stderr[:1000])
