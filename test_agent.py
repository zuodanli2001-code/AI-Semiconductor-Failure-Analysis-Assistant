
from agent import SupervisorAgent


agent = SupervisorAgent()

# 读取 SEM 图片
image_path = r"F:\SEM分析\SEM1.png"

with open(image_path, "rb") as f:
    image_bytes = f.read()


# 测试 1：有图片，要求分析 SEM
print("\n========== 测试 1：SEM 图像分析 ==========")

result1 = agent.run(
    user_input="请分析这张 SEM 图像，检查是否存在明显缺陷。",
    image_bytes=image_bytes
)

print(result1)


# 测试 2：没有图片，只问专业知识
print("\n========== 测试 2：专业知识问答 ==========")

result2 = agent.run(
    user_input="SEM 图像中的裂纹通常有哪些形貌特征？",
    image_bytes=None
)

print(result2)


# 测试 3：有图片，但问无关问题
print("\n========== 测试 3：无关问题 ==========")

result3 = agent.run(
    user_input="请介绍一下 Python 的基本用途。",
    image_bytes=image_bytes
)

print(result3)