"""Static offscreen preview only; never starts a flicker or input loop."""
from pathlib import Path
import os
import json
os.environ['QT_QPA_PLATFORM']='offscreen'
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage,QPainter,QColor,QFont,QFontDatabase
from PyQt5.QtCore import Qt,QRect

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'runs/pilot_virtual_13_nonreal_acceptance_v3'
flow=json.loads((folder/'virtual_flow.json').read_text(encoding='utf-8'))
app=QApplication([])
font_id=QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
assert font_id>=0
font_family=QFontDatabase.applicationFontFamilies(font_id)[0]
image=QImage(1560,730,QImage.Format_RGB32);image.fill(QColor('#f1f5fb'))
p=QPainter(image);p.setRenderHint(QPainter.Antialiasing)
def text(rect,value,size=18,color='#16324f',bold=False):
    p.setPen(QColor(color));p.setFont(QFont(font_family,size,weight=QFont.Bold if bold else QFont.Normal))
    p.drawText(QRect(*rect),Qt.TextWordWrap|Qt.AlignVCenter,value)
text((48,24,1460,66),'步骤13 · 三模式交互界面预览',28,bold=True)
text((48,95,1460,44),'纯软件模拟 ｜ 静态预览 ｜ 未连接脑机或机器人 ｜ 未呈现闪烁刺激',16,color='#58718b')
details=[('TASK_QUERY','01 任务选择','目标选择菜单；响应只更新联合信念。\n\n真实意图不能从算法预测反推标签。'),
    ('CHANNEL_PROBE','02 通道探测','显示已知目标提示，探测输入通道。\n\n探测不是用户任务，不生成执行契约。'),
    ('CONFIRM','03 任务确认','显示具体任务契约和接受／拒绝选项。\n\n确认后仍须通过版本、时效和授权检查。')]
for i,(mode,title,body) in enumerate(details):
    x=48+i*500;p.setBrush(QColor('white'));p.setPen(QColor('#c8d6e5'));p.drawRoundedRect(x,172,464,390,18,18)
    text((x+24,190,416,48),title,23,bold=True)
    text((x+24,247,416,44),mode,14,color='#2372bb')
    text((x+24,300,416,200),body,18)
    text((x+24,508,416,30),'无执行权限 · 请求与菜单版本绑定',13,color='#657d95')
text((48,596,1460,54),'软件闭环：任务选择 → 通道探测 → 契约确认 → 授权 → Fake VLA → 独立虚拟终点验证',18,bold=True)
text((48,657,1460,42),'已验证软件流程与日志重放；真实理解、遵从、解码准确率、屏幕时序及用户耗时尚未验证。',15,color='#657d95')
p.end();assert image.save(str(folder/'ui_preview.png'))
print(str(folder/'ui_preview.png'))
