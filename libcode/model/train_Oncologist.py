import subprocess
#获取当前路径
import os
import sys
root_path = os.getcwd()
#对root_path进行处理,后退两级
root_path = os.path.abspath(os.path.dirname(root_path) + os.path.sep + ".")
def train_Oncologist_step(path):
    # 创建一个新的子进程来执行指定的 shell 命令
    p = subprocess.Popen(["sh", os.path.join(root_path, path)], stdout=subprocess.PIPE)
    # 等待子进程完成
    p.wait()
    # 获取子进程的输出
    output = p.communicate()[0]
    # 打印输出
    print(output)

def main():

    #进行第一步训练pt
    path = "scripts/run_pt.sh"
    train_Oncologist_step(path)
    #进行第二步训练sft
    path = "scripts/run_sft.sh"
    train_Oncologist_step(path)
    #进行第三步训练dpo
    path = "scripts/run_dpo.sh"
    train_Oncologist_step(path)

if __name__ == "__main__":
    main()
