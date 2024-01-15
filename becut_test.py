from bcut_asr import BcutASR
from bcut_asr.orm import ResultStateEnum

asr = BcutASR("./wavs/Erwin_0.wav")
asr.upload() # 上传文件
asr.create_task() # 创建任务

# 轮询检查结果
while True:
    result = asr.result()
    # 判断识别成功
    if result.state == ResultStateEnum.COMPLETE:
        break

# 解析字幕内容
subtitle = result.parse()

# 判断是否存在字幕
if subtitle.has_data():
    # 输出srt格式
    print(subtitle.to_txt())


