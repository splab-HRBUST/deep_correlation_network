# author: muzhan
import os
import sys
import time
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    # print("\ngpu_status0 = ",gpu_status[2])
    # print("gpu_status1 = ",gpu_status[6])
    # print("gpu_status2 = ",gpu_status[10])
    # print("gpu_status3 = ",gpu_status[14])
    # print("gpu_status4 = ",gpu_status[18])
    # print("gpu_status5 = ",gpu_status[22])
    # print("gpu_status6 = ",gpu_status[26])
    # print("gpu_status7 = ",gpu_status[30])
    gpu_memorys = []
    for i in range(8):
        gpu_memory = int(gpu_status[2+4*i].split('/')[0].split('M')[0].strip())
        gpu_memorys.append(gpu_memory)
    # gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    # print("gpu_memorys = ",gpu_memorys)
    return gpu_memorys
def get_idx(list_):
    str1 = []
    for i in list_:
        if i<5000:#设置抢夺gpu的阈值,数值表示gpu已占用的显存
            print("i = ",i," --",str(list_.index(i)))
            str1.append(str(list_.index(i)))
            list_[list_.index(i)] = -1
    print("str1 = ",str1)
    return str1


def narrow_setup(interval=1):
    i = 0
    while len(get_idx(gpu_info()))<1:  # 设置最少使用gpu个数
        gpu_memorys = gpu_info()
        gpu_memory = min(gpu_memorys)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    
    device_ids_list = get_idx(gpu_info())
    print("device_ids_list = ",device_ids_list)
    max_num = 8 #设置gpu个数
    if len(device_ids_list)>max_num-1:
        device_ids_list = device_ids_list[0:]
    device_num = len(device_ids_list)
    print("device_num = ",device_num)
    device_ids = ",".join(device_ids_list)
    print("device_ids = ",str(device_ids))
    # cmd = "python -m torch.distributed.launch --nproc_per_node="+str(device_num)+" main.py  --track=LA --loss=WCE --algo=5 --lr=0.0001 --batch_size=32 --device_ids="+str(device_ids)+" > test_ResNet18_cqt.txt"
    eval_device=str(device_ids[-1])
    # cmd = "python -m torch.distributed.launch --nproc_per_node="+str(1)+" main.py --batch_size=32 --track=LA --loss=WCE --is_eval --eval --device_ids="+str(eval_device)+" --model_path=models/model_cqt_ResNet18_LA_WCE_100_32_0.0001/epoch_99.pth --eval_output=eval_CM_scores_file_cqt_ResNet18.txt >  test_ResNet18_cqt_eval.txt"
    cmd = "python -u main_test2.py --num_epochs=100 --track=logical --features1=spect --features2=mgd --gpu_id="+str(eval_device)+" >em2_7.txt"
    del eval_device
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    # while True:
    narrow_setup()

'''
修改重定向文件名，
main.py中model存储路径名
修改循环轮数
'''