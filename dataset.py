import os 
import shutil
# os.makedirs('rgb')
# for file in sorted(os.listdir('input_images_body/ir')):
#     flag=0
#     for f in sorted(os.listdir('final_json')):
#         if f.split(".")[0]==file.split(".")[0]:
#             flag=1
#             shutil.move(f'final_json/{f}',f'input_images_body/rgb/{f}')
#     if not flag:
#         print(file)
for file in os.listdir('input_images_body/rgb'):
    if file.split(".")[1]=='json':
        shutil.move(f'input_images_body/rgb/{file}',f'input_images_body/json/{file}')
