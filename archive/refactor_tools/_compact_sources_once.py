"""Remove excessive blank lines while proving Python syntax trees unchanged."""
import ast
import io
import tokenize
from pathlib import Path

root = Path(__file__).resolve().parents[1]
names = ['camera_worker.py','body_detector.py','annotator.py','reid_manager.py','body_reid.py']
for name in names:
    path = root/'python_engine'/name
    source = path.read_text(encoding='utf-8-sig')
    lines = source.splitlines(keepends=True)
    protected = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING:
            protected.update(range(token.start[0],token.end[0]+1))
    output = []
    for i,line in enumerate(lines):
        if not line.strip() and i+1 not in protected:
            previous = next((value for value in reversed(output) if value.strip()),'')
            following = next((value for value in lines[i+1:] if value.strip()),'')
            if (previous[:1].isspace() and following[:1].isspace()
                    and not following.lstrip().startswith(('#','def ','async def ','class ','@'))):
                continue
        output.append(line)
    result = ''.join(output)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(source)),name
    path.write_text(result,encoding='utf-8')
    print(f'{name}: {len(lines)} -> {len(output)} lines; syntax tree unchanged')
