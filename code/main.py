import videoprocess as vp
import trainingsetprocess as tp
import videocapture as vc

if __name__ == '__main__':
    tp.trainset_process()
    while True:
        menu = int(input("请输入检测模式（数字）：1. 从本地导入视频检测\t2. 调用摄像头检测\n"))
        if menu == 1:
            vp.video_process()
            break
        elif menu == 2:
            vc.process()
            break
        else:
            print("输入错误，请重新输入！")
            continue




