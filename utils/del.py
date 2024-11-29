import os
for file in os.listdir('test_model'):
    file1,_=os.path.splitext(file)
    print(file1)
    for f in os.listdir('json'):
        if file1 in f:
            os.rename(os.path.join('json',f),os.path.join('json',file1+'.json'))
   