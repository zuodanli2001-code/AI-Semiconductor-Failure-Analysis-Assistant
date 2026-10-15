from sem_analyzer import analyze_sem_image


# 修改成你的 SEM 图片实际路径
image_path = r"E:\Graduate\AgBiSb数据+图片\图片\0%Sb.png"


# 读取图片
with open(image_path, "rb") as f:
    image_bytes = f.read()


# 调用 SEM Analyzer
result = analyze_sem_image(image_bytes)


# 输出结果
print(result)
