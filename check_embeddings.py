import sys, os, numpy as np
sys.path.insert(0, '.')

emb_dir = 'data/embeddings'
stored  = {}
names   = {}

for f in os.listdir(emb_dir):
    if f.endswith('.npy'):
        sid = f.replace('.npy','')
        e   = np.load(os.path.join(emb_dir, f))
        stored[sid] = e / np.linalg.norm(e)

with open(os.path.join(emb_dir, 'names.txt'), 'r') as f:
    for line in f:
        if ',' in line:
            sid, name = line.strip().split(',',1)
            names[sid.strip()] = name.strip()

print('Inter-student similarity matrix:')
print()

sids = list(stored.keys())
header = '               ' + ''.join(f'{names.get(s,s):15s}' for s in sids)
print(header)

for s1 in sids:
    row = f'{names.get(s1,s1):15s}'
    for s2 in sids:
        sim = float(np.dot(stored[s1], stored[s2]))
        if s1 == s2:
            row += f'{"1.000":15s}'
        else:
            row += f'{sim:<15.4f}'
    print(row)

print()
print('Threshold recommendation:')
max_inter = 0
confused  = []
for i, s1 in enumerate(sids):
    for j, s2 in enumerate(sids):
        if i >= j:
            continue
        sim = float(np.dot(stored[s1], stored[s2]))
        if sim > max_inter:
            max_inter = sim
        if sim > 0.35:
            confused.append((names.get(s1,s1), names.get(s2,s2), sim))

print(f'Max inter-student similarity: {max_inter:.4f}')
print(f'Recommended threshold: {max_inter + 0.10:.4f}')
if confused:
    print('Students that may be confused:')
    for a, b, s in confused:
        print(f'  {a} vs {b}: {s:.4f}')
else:
    print('No students above 0.35 similarity - embeddings are distinct')
