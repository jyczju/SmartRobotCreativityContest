def compare(red_qizi, green_qizi):

    jqDict = {
    'gongbin'   : 0,
    'paizhang'  : 1,
    'lianzhang' : 2,
    'yinzhang'  : 3,
    'tuanzhang' : 4,
    'lvzhang'   : 5,
    'shizhang'  : 6,
    'junzhang'  : 7,
    'siling'    : 10,               ####大子的话，把司令和军长算入就好了吧！
    'dilei'     : 20,
    'zhadan'    : 30,
    'junqi'     : -1
    }
    ######返回值又加了一个，显示哪方赢了，用于统计战损比,'red'、'green'、'red_green'
    ######之后返回值还加了一个，显示大子和炸弹利用率：'small'、'red_large'、'green_large'、'red_bang'、'green_bang' 
    #返回的三个值分别为：“屏幕显示结果”、“战损比统计（返回赢家）”、“大子和炸弹利用率统计”   
    if jqDict[red_qizi] == -1:                      
        return 'green_win','green','small' # A      
    if jqDict[green_qizi] == -1:
        return 'red_win','red','small' # B
    if jqDict[red_qizi] == jqDict[green_qizi] and jqDict[red_qizi] == 10:
        return 'equal_siling','red_green','small' # C
    if jqDict[red_qizi] == 10 and jqDict[green_qizi] == 30:
        return 'red_siling_killed_by_zhadan','red_green','green_bang' # D
    if jqDict[red_qizi] == 30 and jqDict[green_qizi] == 10:
        return 'green_siling_killed_by_zhadan','red_green','red_bang' # E
    if jqDict[red_qizi] == 30:
        return 'equal','red_green','red_bang' # F
    if jqDict[green_qizi] == 30:
        return 'equal','red_green','green_bang' # F
    if jqDict[red_qizi] == jqDict[green_qizi]:
        return 'equal','red_green','small' # F
    if jqDict[red_qizi] == 10 and jqDict[green_qizi] == 20:
        return 'red_siling_killed_by_dilei','red_green','small' # G
    if jqDict[red_qizi] == 20 and jqDict[green_qizi] == 10:
        return 'green_siling_killed_by_dilei','red_green','small' # H
    if jqDict[red_qizi] == 0 and jqDict[green_qizi] == 20:
        return 'red_kill_green','red','small' # I
    if jqDict[red_qizi] == 20 and jqDict[green_qizi] == 0:
        return 'green_kill_red','green','small' # J
    if jqDict[red_qizi] > jqDict[green_qizi]:
        if jqDict[red_qizi]>=7 and jqDict[red_qizi]<=10:
                return 'red_kill_green','red','red_large' # K
        else:
            return 'red_kill_green','red','small' # K
    if jqDict[red_qizi] < jqDict[green_qizi]:
        if jqDict[green_qizi]>=7 and jqDict[green_qizi]<=10:
            return 'green_kill_red','green','green_large' # L
        else:
            return 'green_kill_red','green','small'
    
if __name__ == '__main__':
    red_qizi = 'siling'
    green_qizi = 'siling'
    result = compare(red_qizi, green_qizi)
    print(result)
