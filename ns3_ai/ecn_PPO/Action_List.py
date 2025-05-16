
def Action():
    kmin = [2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]#13
    kmax = [128, 256, 512, 1024, 2048, 5120, 10240]#7
    pmax = [0.01, 0.1, 0.25, 0.5, 0.75, 1]#6
    action_space = [[0] * 3] * 390
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space

def Action1():
    kmin = [2, 5, 10, 20, 40, 80, 160, 320, 640]
    kmax = [128, 256, 512, 1024, 2048]
    pmax = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
    action_space = [[0] * 3] * 234
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space,num

def Action2():
    kmin = [2, 4, 8, 16, 32]
    kmax = [16, 32, 64, 128, 256]
    pmax = [0.001, 0.25, 0.5, 0.75, 1]
    action_space = [[0] * 3] * 120
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space,num


def Action3():#160
    kmin = [5, 10, 20, 40, 80, 160, 320]
    kmax = [80,160,320,640,1280]
    pmax = [0.001, 0.25, 0.5, 0.75, 1]
    action_space = [[0] * 3] * 160
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space