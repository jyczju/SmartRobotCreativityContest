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
    'siling'    : 10,
    'dilei'     : 20,
    'zhadan'    : 30,
    'junqi'     : -1
    }

    if jqDict[red_qizi] == -1:
        return 'green_win'
    if jqDict[green_qizi] == -1:
        return 'red_win'
    if jqDict[red_qizi] == jqDict[green_qizi] and jqDict[red_qizi] == 10:
        return 'equal_siling'
    if jqDict[red_qizi] == 10 and jqDict[green_qizi] == 30:
        return 'red_siling_killed_by_zhadan'
    if jqDict[red_qizi] == 30 and jqDict[green_qizi] == 10:
        return 'green_siling_killed_by_zhadan'
    if jqDict[red_qizi] == 30 or jqDict[green_qizi] == 30 or jqDict[red_qizi] == jqDict[green_qizi]:
        return 'equal'
    if jqDict[red_qizi] == 10 and jqDict[green_qizi] == 20:
        return 'red_siling_killed_by_dilei'
    if jqDict[red_qizi] == 20 and jqDict[green_qizi] == 10:
        return 'green_siling_killed_by_dilei'
    if jqDict[red_qizi] == 0 and jqDict[green_qizi] == 20:
        return 'red_kill_green'
    if jqDict[red_qizi] == 20 and jqDict[green_qizi] == 0:
        return 'green_kill_red'
    if jqDict[red_qizi] > jqDict[green_qizi]:
        return 'red_kill_green'
    if jqDict[red_qizi] < jqDict[green_qizi]:
        return 'green_kill_red'
    
if __name__ == '__main__':
    red_qizi = 'siling'
    green_qizi = 'siling'
    result = compare(red_qizi, green_qizi)
    print(result)
