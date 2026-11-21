#!/usr/bin/env python3
"""ResNet-18 baseline: train only on labeled training images; predict in template order."""
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
MEAN=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
STD=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]

class Block(nn.Module):
    expansion=1
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False);self.bn1=nn.BatchNorm2d(cout)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False);self.bn2=nn.BatchNorm2d(cout)
        self.downsample=nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout)) if stride!=1 or cin!=cout else None
    def forward(self,x):
        identity=x if self.downsample is None else self.downsample(x)
        x=self.relu(self.bn1(self.conv1(x)));x=self.bn2(self.conv2(x))
        return self.relu(x+identity)

class ResNet18(nn.Module):
    """Standard ResNet-18 layout, compatible with official PyTorch ImageNet weights."""
    def __init__(self,n=1000):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,7,2,3,bias=False);self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True);self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=nn.Sequential(Block(64,64),Block(64,64))
        self.layer2=nn.Sequential(Block(64,128,2),Block(128,128))
        self.layer3=nn.Sequential(Block(128,256,2),Block(256,256))
        self.layer4=nn.Sequential(Block(256,512,2),Block(512,512))
        self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.fc=nn.Linear(512,n)
    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x),1))

def device_for(name):
    if name!='auto':return torch.device(name)
    if torch.cuda.is_available():return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def transform(path,size,augment):
    with Image.open(path) as original:
        im=ImageOps.exif_transpose(original).convert('RGB')
        if augment:
            if random.random()<.5:im=ImageOps.mirror(im)
            im=im.rotate(random.uniform(-10,10),resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.85,1.15))
            im=ImageEnhance.Contrast(im).enhance(random.uniform(.85,1.15))
        # Preserve the complete frame, including small/off-center characters.
        im=ImageOps.pad(im,(size,size),method=Image.Resampling.BILINEAR,color=(128,128,128))
        a=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.
    return torch.from_numpy((a-MEAN)/STD)

class Images(Dataset):
    def __init__(self,items,size,augment=False):self.items=items;self.size=size;self.augment=augment
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        try:return transform(p,self.size,self.augment),y
        except Exception as e:raise RuntimeError(f'Cannot decode image: {p}') from e

def train_root(path):
    p=Path(path)
    for _ in range(5):
        dirs=sorted(x for x in p.iterdir() if x.is_dir() and not x.name.startswith('.'))
        if len(dirs)==1:p=dirs[0]
        else:return p
    raise ValueError('Too many nested folders; supply directory containing character folders')

def loader(ds,args,shuffle=False):
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.workers,pin_memory=args.device.type=='cuda',generator=torch.Generator().manual_seed(args.seed))

def train(args):
    seed_all(args.seed);args.device=device_for(args.device);torch.set_num_threads(args.threads)
    if args.epochs<1 or not 0<args.val_fraction<1:raise ValueError('Invalid epochs / validation fraction')
    root=train_root(args.train_dir);classes=sorted(x.name for x in root.iterdir() if x.is_dir() and not x.name.startswith('.'))
    if len(classes)!=50:raise ValueError(f'Expected 50 character folders; found {len(classes)} at {root}')
    rng=random.Random(args.seed);tr=[];va=[]
    for y,c in enumerate(classes):
        files=sorted(p for p in (root/c).rglob('*') if p.is_file() and p.suffix.lower() in EXTS)
        rng.shuffle(files)
        if args.max_per_class:files=files[:args.max_per_class]
        if len(files)<2:raise ValueError(f'{c}: need at least 2 images')
        nv=max(1,min(len(files)-1,round(len(files)*args.val_fraction)))
        va.extend((p,y) for p in files[:nv]);tr.extend((p,y) for p in files[nv:])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Use a fresh output directory to avoid overwriting a previous run')
    with (out/'split.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','character','split'])
        for split,items in [('train',tr),('validation',va)]:
            w.writerows((str(p.relative_to(root)),classes[y],split) for p,y in items)
    model=ResNet18()
    if not args.scratch:
        weights=torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',map_location='cpu',check_hash=True)
        model.load_state_dict(weights,strict=True)
    model.fc=nn.Linear(512,len(classes));model.to(args.device)
    optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('fc.')],'lr':args.lr},{'params':model.fc.parameters(),'lr':args.lr*10}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    criterion=nn.CrossEntropyLoss();best=-1;stale=0;history=[]
    tl=loader(Images(tr,args.image_size,True),args,True);vl=loader(Images(va,args.image_size),args)
    smoke=bool(args.max_batches or args.max_per_class)
    print(json.dumps({'device':str(args.device),'train':len(tr),'validation':len(va),'classes':len(classes),'smoke_or_subset':smoke}),flush=True)
    for epoch in range(1,args.epochs+1):
        start=time.time();model.train();loss_sum=0.;seen=0
        for bi,(x,y) in enumerate(tl):
            x=x.to(args.device);y=y.to(args.device);optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x),y);loss.backward();optimizer.step();loss_sum+=loss.item()*len(y);seen+=len(y)
            if args.max_batches and bi+1>=args.max_batches:break
        model.eval();pred=[];true=[]
        with torch.inference_mode():
            for bi,(x,y) in enumerate(vl):
                pred.extend(model(x.to(args.device)).argmax(1).cpu().tolist());true.extend(y.tolist())
                if args.max_batches and bi+1>=args.max_batches:break
        correct=np.array(pred)==np.array(true);acc=float(correct.mean())
        per_class={c:float(correct[np.array(true)==i].mean()) if i in true else None for i,c in enumerate(classes)}
        row={'epoch':epoch,'train_loss':loss_sum/seen,'validation_accuracy':acc,'validation_evaluated':len(true),'seconds':time.time()-start,'smoke_or_subset':smoke};history.append(row);print(json.dumps(row),flush=True)
        if acc>best:
            best=acc;stale=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'classes':classes,'image_size':args.image_size,'epoch':epoch,'validation_accuracy':acc,'seed':args.seed,'smoke_or_subset':smoke},out/'best.pt')
            (out/'validation_by_class.json').write_text(json.dumps(per_class,indent=2))
        else:stale+=1
        (out/'history.json').write_text(json.dumps(history,indent=2));scheduler.step()
        if stale>=args.patience:break
    config={k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items() if k!='fn'}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    print('Saved:',out/'best.pt',flush=True)

def predict(args):
    seed_all(args.seed);args.device=device_for(args.device);torch.set_num_threads(args.threads)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('smoke_or_subset'):print('WARNING: checkpoint is from a smoke/subset run, not a full baseline.',flush=True)
    classes=checkpoint['classes'];model=ResNet18(len(classes));model.load_state_dict(checkpoint['state_dict']);model.to(args.device).eval()
    with Path(args.sample_csv).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','character']:raise ValueError('Template must have exactly id,character columns')
        rows=list(reader)
    ids=[r['id'] for r in rows]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Template IDs are empty or duplicated')
    images={}
    for p in Path(args.test_dir).rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            if p.stem in images:raise ValueError(f'Duplicate image ID: {p.stem}')
            images[p.stem]=p
    if set(ids)!=set(images):raise ValueError(f'Template/image mismatch: missing={len(set(ids)-set(images))}, extra={len(set(images)-set(ids))}')
    ds=Images([(images[i],0) for i in ids],checkpoint['image_size']);pred=[]
    with torch.inference_mode():
        for x,_ in loader(ds,args):pred.extend(model(x.to(args.device)).argmax(1).cpu().tolist())
    assert len(pred)==len(ids)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['id','character']);w.writerows((i,classes[y]) for i,y in zip(ids,pred))
    print(f'Saved {len(ids)} predictions: {path}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    def common(s):
        s.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps']);s.add_argument('--batch-size',type=int,default=32)
        s.add_argument('--workers',type=int,default=0);s.add_argument('--threads',type=int,default=4);s.add_argument('--seed',type=int,default=2026)
    t=sub.add_parser('train');common(t);t.add_argument('--train-dir',required=True);t.add_argument('--output',default='runs/baseline')
    t.add_argument('--epochs',type=int,default=10);t.add_argument('--patience',type=int,default=3);t.add_argument('--lr',type=float,default=1e-4)
    t.add_argument('--val-fraction',type=float,default=.2);t.add_argument('--image-size',type=int,default=224)
    t.add_argument('--scratch',action='store_true',help='Diagnostic only: skip pretrained weights')
    t.add_argument('--max-per-class',type=int,default=0,help='Smoke/subset test only');t.add_argument('--max-batches',type=int,default=0,help='Smoke test only');t.set_defaults(fn=train)
    s=sub.add_parser('predict');common(s);s.add_argument('--checkpoint',required=True);s.add_argument('--test-dir',required=True);s.add_argument('--sample-csv',required=True);s.add_argument('--output',default='submission.csv');s.set_defaults(fn=predict)
    args=p.parse_args();args.fn(args)
if __name__=='__main__':main()
