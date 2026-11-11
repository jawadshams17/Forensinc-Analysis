
import sys
from volatility3.cli import main
sys.argv = [
    "vol.py",
    "-f", r"F:\FORENSICS\adversarial-memforensics\data\malvol25\tmp\Cerber-NoVirus.mem",
    "-r", "json",
    "windows.pslist.PsList"
]
main()
