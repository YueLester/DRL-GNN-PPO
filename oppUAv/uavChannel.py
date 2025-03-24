import  time

start_time = time.time()



for i in range(1,2000):
    for j in range(1, 2000):
        for k in range(1, 2000):
            a = 2
            # print(i,j,k)

end_time = time.time()

# 计算并打印执行时间
execution_time = end_time - start_time
print("执行时间: {"+str(execution_time)+":.6f} 秒")