import os
import re

md_path = r'C:\Users\Microsoft\.gemini\antigravity\brain\11a9adb3-edfa-42c0-9245-f3b6f60f73a8\walkthrough.md'
with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('````carousel', '').replace('````', '').replace('<!-- slide -->', '\n\n---\n\n')

out_md = r'f:\FORENSICS\adversarial-memforensics\results\unified\walkthrough_print.md'
with open(out_md, 'w', encoding='utf-8') as f:
    f.write(text)
