import matplotlib.pyplot as plt
'''
x_list = [0.6,2.6,4.6,6.6,8.6,10.6,12.6]
y_list = [4.9965,2.7837,4.4521,6.2335,7.5584,9.9285,11.1219]
x_list1 = [0.8,2.8,4.8,6.8,8.8,10.8,12.8]
y_list1 = [5.6579,3.1712,5.0023,7.0214,8.4177,11.0752,12.6124]
x_list2 = [1,3,5,7,9,11,13]
y_list2 = [1.5742,1.5997,2.757,4.136,5.2362,6.4775,7.573]
x_list3 = [1.2,3.2,5.2,7.2,9.2,11.2,13.2]
y_list3 = [3.3889,1.6523,3.0056,4.3467,5.5611,6.8258,7.9484]
x_list4 = [1.4,3.4,5.4,7.4,9.4,11.4,13.4]
y_list4 = [3.4757,1.6487,3.086,4.5502,5.7643,7.0186,8.2993]
plt.bar(x_list, y_list, color='b', width=0.2)
plt.bar(x_list1, y_list1, color='r', width=0.2)
plt.bar(x_list2, y_list2, color='c', tick_label=["auto","Exit 1","Exit 2","Exit 3","Exit 4","Exit 5","Exit 6"],width=0.2 )#,align='edge'
plt.bar(x_list3, y_list3, color='g', width=0.2)
plt.bar(x_list4, y_list4, color='m', width=0.2)

plt.xlabel("Exit")
plt.ylabel("Time(ms)")
plt.title("Time using of each exit")
for i in range(len(x_list)):
    plt.text(x_list[i], y_list[i], "{:.1f}".format(y_list[i]), ha='center', va='bottom')
    plt.text(x_list1[i], y_list1[i], "{:.1f}".format(y_list1[i]), ha='center', va='bottom')
    plt.text(x_list2[i], y_list2[i], "{:.1f}".format(y_list2[i]), ha='center', va='bottom')
    plt.text(x_list3[i], y_list3[i], "{:.1f}".format(y_list3[i]), ha='center', va='bottom')
    plt.text(x_list4[i], y_list4[i], "{:.1f}".format(y_list4[i]), ha='center', va='bottom')
plt.show()'''

x_list = [0.6,2.6,4.6,6.6,8.6,10.6]
y_list = [2.7837,1.7241,1.7886,1.3787,2.27,1.3989]
x_list1 = [0.8,2.8,4.8,6.8,8.8,10.8]
y_list1 = [3.1712,1.8474,2.0118,1.4797,2.6631,1.5735]
x_list2 = [1,3,5,7,9,11]
y_list2 = [1.5997,1.2229,1.2977,1.1681,1.1908,1.1245]
x_list3 = [1.2,3.2,5.2,7.2,9.2,11.2]
y_list3 = [1.6523,1.3687,1.3914,1.2276,1.2747,1.1254]
x_list4 = [1.4,3.4,5.4,7.4,9.4,11.4]
y_list4 = [1.6487,1.4073,1.4724,1.2786,1.2813,1.2078]
plt.bar(x_list, y_list, color='b', width=0.2)
plt.bar(x_list1, y_list1, color='r', width=0.2)
plt.bar(x_list2, y_list2, color='c', tick_label=["C1","S2","C3","S4","F5","Final Output"],width=0.2 )#,align='edge'
plt.bar(x_list3, y_list3, color='g', width=0.2)
plt.bar(x_list4, y_list4, color='m', width=0.2)

plt.xlabel("Exit")
plt.ylabel("Time(ms)")
plt.title("Time using of each layer")
for i in range(len(x_list)):
    plt.text(x_list[i], y_list[i], "{:.1f}".format(y_list[i]), ha='center', va='bottom')
    plt.text(x_list1[i], y_list1[i], "{:.1f}".format(y_list1[i]), ha='center', va='bottom')
    plt.text(x_list2[i], y_list2[i], "{:.1f}".format(y_list2[i]), ha='center', va='bottom')
    plt.text(x_list3[i], y_list3[i], "{:.1f}".format(y_list3[i]), ha='center', va='bottom')
    plt.text(x_list4[i], y_list4[i], "{:.1f}".format(y_list4[i]), ha='center', va='bottom')
plt.show()