import re

with open('data.py') as f:
    content = f.read()

new_fn = '''def sample_all_types(dset, min_k=5, max_attempts=200):
    total_types = []
    for i in dset.index:
        types = list(set([miniproc(x) for x in dset.loc[i, "exact_types"]]))
        total_types.extend(types)
    total_types = list(set(total_types))
    done = False
    k = min_k
    i = 0
    minidset = None
    best_minidset = None
    best_coverage = -1
    while not done:
        selected_types = []
        k = min(k, len(dset))
        minidset = dset.sample(k).reset_index(drop=True)
        for j in minidset.index:
            types = list(set([miniproc(x) for x in minidset.loc[j, "exact_types"]]))
            selected_types.extend(types)
        selected_types = list(set(selected_types))
        coverage = len(selected_types)
        if coverage > best_coverage:
            best_coverage = coverage
            best_minidset = minidset
        if coverage == len(total_types):
            done = True
            break
        i += 1
        if (i + 1) % 10 == 0:
            k += 1
        if i >= max_attempts:
            print(f"sample_all_types: gave up after {max_attempts} attempts, "
                  f"using best partial sample ({best_coverage}/{len(total_types)} types covered)")
            minidset = best_minidset
            done = True
    return minidset
'''

pattern = re.compile(r'^def sample_all_types\(.*?\n(?=^def )', re.DOTALL | re.MULTILINE)
new_content, n = pattern.subn(new_fn + chr(10), content, count=1)

if n == 0:
    print('ERROR: could not find sample_all_types() to replace.')
else:
    with open('data.py', 'w') as f:
        f.write(new_content)
    print(f'Replaced sample_all_types() successfully ({n} replacement made).')
