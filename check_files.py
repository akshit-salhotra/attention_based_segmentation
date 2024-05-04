import os 
for file in os.listdir('input_images_body/rgb'):
    flag=0
    for f in os.listdir('input_images_body/ir'):
        if f.split(".")[0]==file.split(".")[0]:
            flag=1
            break
    if not flag:
        print(f'input_images_body/rgb/{file}')
        # os.remove(f"input_images_body/json/{file}")