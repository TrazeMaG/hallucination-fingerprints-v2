python -c "
f = open('experiments/experiment_18_patching_dola.py', 'rb').read()
print('Size:', len(f), 'bytes')
print('First 50 bytes:', f[:50])
print('Has BOM:', f[:3] == b'\xef\xbb\xbf')
"