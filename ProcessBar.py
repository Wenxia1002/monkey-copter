#-- coding:UTF-8 --
import sys
import time


def progress_bar(finish_tasks_number, tasks_number):
    """
    进度条

    :param finish_tasks_number: int, 已完成的任务数
    :param tasks_number: int, 总的任务数
    :return:
    """

    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    sys.stdout.flush()


if __name__ == '__main__':
    for i in range(0, 101):
        progress_bar(i, 100)
        time.sleep(0.05)
