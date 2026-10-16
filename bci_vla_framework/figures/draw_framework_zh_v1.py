"""Localize the editable source drawing; preserve the English original.

Only string literals and the font change. Geometry, colors, arrows and scope
annotations are inherited exactly from the previously reviewed vector source.
"""
import ast
import json
from pathlib import Path
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
for path in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/msyhbd.ttc']:
    font_manager.fontManager.addfont(path)

labels=json.loads((ROOT/'framework_zh_v1_labels.json').read_text(encoding='utf-8'))
outputs={'bci_vla_framework_v1.svg':'bci_vla_framework_zh_v1.svg',
         'bci_vla_framework_v1.png':'bci_vla_framework_zh_v1.png'}
seen=set()

class Localize(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            if node.value in labels:
                seen.add(node.value)
                return ast.copy_location(ast.Constant(labels[node.value]),node)
            if node.value in outputs:
                return ast.copy_location(ast.Constant(outputs[node.value]),node)
            if node.value=='Arial':
                return ast.copy_location(ast.Constant('Microsoft YaHei'),node)
        return node

tree=Localize().visit(ast.parse((ROOT/'draw_framework_v1.py').read_text(encoding='utf-8')))
ast.fix_missing_locations(tree)
if set(labels)-seen:raise ValueError('Unused labels: '+repr(set(labels)-seen))
exec(compile(tree,str(ROOT/'draw_framework_v1.py'),'exec'),{'__file__':str(ROOT/'draw_framework_v1.py'),'__name__':'__main__'})
print(f'Localized {len(seen)} labels; original English SVG and PNG preserved.')
