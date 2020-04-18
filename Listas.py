#Intento de organizar Train, Val y Test
import glob
import os
todos=glob.glob(os.path.join('*','*','*','*.nii.gz'))
if len(todos) == 0:
    todos=glob.glob(os.path.join('*','*','*','*.gz'))
test=[]
val=[]
train=[]
segtr=[]
segv=[]
segte=[]
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and len(train)<520 and todos[i].find('seg')==-1:
        train.append(todos[i])
        print(i)
    if todos[i].find('LGG')!=-1 and len(train)<672 and todos[i].find('seg')==-1:
        train.append(todos[i])
        print(i)
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and len(val)<256 and todos[i].find('seg')==-1:
        if any(todos[i] in string for string in train)== False:
            val.append(todos[i])
            print(i)
    if todos[i].find('LGG')!=-1 and len(val)<332 and todos[i].find('seg')==-1:
        if any(todos[i] in string for string in train)== False:
            val.append(todos[i])
            print(i)
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and len(test)<260 and todos[i].find('seg')==-1:
        if any(todos[i] in string for string in train)== False and any(todos[i] in string for string in val)==False:
            test.append(todos[i])
            print(i)
    if todos[i].find('LGG')!=-1 and len(train)<336 and todos[i].find('seg')==-1:
        if any(todos[i] in string for string in train)== False and any(todos[i] in string for string in val)==False:
            test.append(todos[i])
            print(i)
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and todos[i].find('seg')!=-1 and len(segtr)<130:
        segtr.append(todos[i])
    if todos[i].find('LGG')!=-1 and todos[i].find('seg')!=-1 and len(segtr)<168:
        segtr.append(todos[i])
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and todos[i].find('seg')!=-1 and len(segv)<64:
        if any(todos[i] in string for string in segtr)==False:
            segv.append(todos[i])
    if todos[i].find('LGG')!=-1 and todos[i].find('seg')!=-1 and len(segv)<83:
        if any(todos[i] in string for string in segtr)==False:
            segv.append(todos[i])
for i in range(0,len(todos)-1):
    if todos[i].find('HGG')!=-1 and todos[i].find('seg')!=-1 and len(segte)<65:
        if any(todos[i] in string for string in segtr)==False and any(todos[i] in string for string in segv)==False:
            segte.append(todos[i])
    if todos[i].find('LGG')!=-1 and todos[i].find('seg')!=-1 and len(segte)<84:
        if any(todos[i] in string for string in segtr)==False and any(todos[i] in string for string in segv)==False:
            segte.append(todos[i])
trpair=[]
cont=0
cont1=0
for i in range(0,len(train)):
    txt=train[i]+' '+segtr[cont]
    trpair.append(txt)
    cont1+=1
    if cont1==4:
        cont+=1
        cont1=0
vpair=[]
cont=0
cont1=0
for i in range(0,len(val)):
    txt=val[i]+' '+segv[cont]
    vpair.append(txt)
    cont1+=1
    if cont1==4:
        cont+=1
        cont1=0
txt = '\n'.join(trpair)
f = open ('train_pair.lst','a')
f.write(txt)
f.close()
txt1= '\n'.join(vpair)
h=open('val_pair.lst','a')
h.write(txt1)
h.close()
txt2= '\n'.join(test)
t=open('test.lst','a')
t.write(txt2)
t.close()
txt3= '\n'.join(segte)
g=open('test_seg.lst','a')
g.write(txt3)
g.close()
txt4= '\n'.join(train)
p=open('train.lst','a')
p.write(txt4)
p.close()
txt5= '\n'.join(segtr)
k=open('train_seg.lst','a')
k.write(txt5)
k.close()
txt6= '\n'.join(val)
n=open('val.lst','a')
n.write(txt6)
n.close()
txt7= '\n'.join(segv)
m=open('val_seg.lst','a')
m.write(txt7)
m.close()