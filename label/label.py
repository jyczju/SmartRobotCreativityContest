import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#列名
# col=[]
# for i in range(1,8):
#     col.append("Day"+str(i))
col = ['红方棋子','绿方棋子','结果']


#行名
# row=[]
# for i in range(1,13):
#     row.append(i)

row = ['第1步','第2步','第3步','第4步','第5步']
#表格里面的具体值
vals=[['团长','连长','red_killed_green'],['司令','工兵','red_killed_green'],['团长','师长','green_killed_red'],['炸弹','军长','equal'],['军旗','工兵','green_win']]

col_colors = ['red','green','gold']
plt.figure()
tab = plt.table(cellText=vals, 
              colLabels=col, 
             rowLabels=row,
             colColours=col_colors,
             colWidths=[0.3]*3,
              loc='center', 
              cellLoc='center',
              rowLoc='center')
tab.scale(1,2) 
plt.axis('off')
plt.show()
