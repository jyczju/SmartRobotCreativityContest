import cv2


red_remaining = 25
green_remaining = 25
img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/normal.jpg')

if (compare_result=='green_win'):
    img = cv2.imread('./green_win.jpg')
    red_remaining = red_remaining - 1
elif (compare_result=='red_win'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/red_win.jpg')
    green_remaining = green_remaining - 1
elif (compare_result=='equal_siling'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/equal_siling.jpg')
    red_remaining = red_remaining - 1
    green_remaining = green_remaining - 1
elif (compare_result=='red_siling_killed_by_zhadan'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/red_siling_killed_by_zhadan.jpg')
    red_remaining = red_remaining - 1
    green_remaining = green_remaining - 1
elif (compare_result=='green_siling_killed_by_zhadan'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/green_siling_killed_by_zhadan.jpg')
    red_remaining = red_remaining - 1
    green_remaining = green_remaining - 1
elif (compare_result=='equal'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/equal.jpg')
    red_remaining = red_remaining - 1
    green_remaining = green_remaining - 1
elif (compare_result=='red_siling_killed_by_dilei'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/red_siling_killed_by_dilei.jpg')
    red_remaining = red_remaining - 1
elif (compare_result=='green_siling_killed_by_dilei'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/green_siling_killed_by_dilei.jpg')
    green_remaining = green_remaining - 1
elif (compare_result=='red_kill_green'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/red_kill_green.jpg')
    green_remaining = green_remaining - 1
elif (compare_result=='green_kill_red'):
    img = cv2.imread('c:/Users/86157/Desktop/University/Miscellaneous/SRCC/green_kill_red.jpg')
    red_remaining = red_remaining - 1

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, str(red_remaining), (140,300), font, 1, (0,0,255), 2, cv2.LINE_AA)
cv2.putText(img, str(green_remaining), (270,300), font, 1, (0,255,0), 2, cv2.LINE_AA)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
