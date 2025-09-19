from PIL import Image


# 打开图片A
with Image.open('results/results_LL/vis_L1.png') as imgA:
    # 顺时针旋转90度
    width, height = imgA.size

    left = width / 4
    right = width - width / 6
    upper = height / 3
    lower = height - height / 3
    imgA = imgA.crop((left, upper, right, lower))
    # 保存处理后的图片A
    imgA.save('results/results_LL/vis_L1.png')
    
with Image.open('results/results_LL/vis_L2.png') as imgB:
    # 顺时针旋转90度
    imgB = imgB.rotate(-90, expand=True)
    # 垂直方向镜像映射
    # imgA_transposed = imgA_rotated.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    width, height = imgB.size

    left = width / 4
    right = width - width / 4
    upper = height / 4
    lower = height - height / 5
    imgB = imgB.crop((left, upper, right, lower))
    # 保存处理后的图片A
    imgB.save('results/results_LL/vis_L2.png')

# 打开图片B
with Image.open('results/results_LL/vis_L1_cop_L2.png') as imgAB:
    # imgAB = imgAB.rotate(-90, expand=True)
    # 水平方向镜像映射
    # imgAB = imgAB.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    width, height = imgAB.size
    left = width / 4
    right = width - width / 6
    
    upper = height / 4
    lower = height - height / 3
    imgAB = imgAB.crop((left, upper, right, lower))

    # 保存处理后的图片B
    imgAB.save('results/results_LL/vis_L1_cop_L2.png')