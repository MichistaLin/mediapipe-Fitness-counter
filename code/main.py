import videoprocess as vp
import trainingsetprocess as tp
import videocapture as vc

if __name__ == '__main__':
    while True:
        menu = int(input("请输入检测模式（数字）：1. 从本地导入视频检测\t2. 调用摄像头检测\t3. 退出\n"))
        if menu == 1:
            flag = int(input("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上（暂未获得csv文件）\n"))
            video_path = input("请输入视频路径：")
            tp.trainset_process(flag)
            vp.video_process(video_path, flag)
            continue
        elif menu == 2:
            flag = int(input("请输入检测的运动类型（数字）：1. 俯卧撑\t2. 深蹲\t3. 引体向上（暂未获得csv文件）\n"))
            print("\n按英文状态下的q或esc退出摄像头采集")
            tp.trainset_process(flag)
            vc.process(flag)
            continue
        elif menu == 3:
            break
        else:
            print("输入错误，请重新输入！")
            continue
