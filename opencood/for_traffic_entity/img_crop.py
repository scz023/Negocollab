from PIL import Image


# 打开图片A
with Image.open('results/result_A.png') as imgA:
    # 顺时针旋转90度
    width, height = imgA.size

    left = width / 3
    right = width - width / 4
    upper = height / 3
    lower = height - height / 3
    imgA = imgA.crop((left, upper, right, lower))
    # 保存处理后的图片A
    imgA.save('results/result_A.png')

# 打开图片A
with Image.open('results/result_B.png') as imgB:
    # 顺时针旋转90度
    imgA_rotated = imgB.rotate(-90, expand=True)
    # 垂直方向镜像映射
    imgA_transposed = imgA_rotated.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    width, height = imgA_transposed.size

    left = width / 3
    right = width - width / 3
    upper = height / 3
    lower = height - height / 4
    image = imgA_transposed.crop((left, upper, right, lower))
    # 保存处理后的图片A
    image.save('results/result_B.png')

# 打开图片B
with Image.open('results/result_A_cop_B.png') as imgAB:
    # 水平方向镜像映射
    imgAB = imgAB.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    width, height = imgAB.size
    left = width / 3
    right = width - width / 4
    upper = height / 3
    lower = height - height / 3
    imgAB = imgAB.crop((left, upper, right, lower))

    # 保存处理后的图片B
    imgAB.save('results/result_A_cop_B.png')
