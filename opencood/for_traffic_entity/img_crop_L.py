from PIL import Image


# 打开图片A
with Image.open('results/results_L/vis_L.png') as imgA:
    # 顺时针旋转90度
    width, height = imgA.size

    left = width / 4
    right = width - width / 6
    upper = height / 3
    lower = height - height / 3
    imgA = imgA.crop((left, upper, right, lower))
    # 保存处理后的图片A
    imgA.save('results/results_L/vis_L.png')