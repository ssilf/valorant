"""Генератор 3D-моделей (.glb) для игры. Запуск: python3 make_models.py
Нужен только numpy. Все размеры в метрах. Меняй цвета и размеры и запускай заново."""
import numpy as np, json, struct, os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'models');os.makedirs(OUT,exist_ok=True)
PI=np.pi

def rot(rx=0,ry=0,rz=0):
    cx,sx,cy,sy,cz,sz=np.cos(rx),np.sin(rx),np.cos(ry),np.sin(ry),np.cos(rz),np.sin(rz)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
def fin(P,N,I,c,r):
    R=rot(*r);P=np.array(P,float)@R.T+np.array(c,float);N=np.array(N,float)@R.T
    I=np.array(I,dtype=np.uint32).reshape(-1,3);a,b,d=P[I[:,0]],P[I[:,1]],P[I[:,2]]
    f=(np.cross(b-a,d-a)*(N[I[:,0]]+N[I[:,1]]+N[I[:,2]])).sum(1)<0
    I[f]=I[f][:,[0,2,1]];return P.astype(np.float32),N.astype(np.float32),I.reshape(-1)

def box(s,c,r=(0,0,0)):
    sx,sy,sz=[v/2 for v in s];P=[];N=[];I=[]
    F=[((1,0,0),[(1,-1,-1),(1,1,-1),(1,1,1),(1,-1,1)]),((-1,0,0),[(-1,-1,1),(-1,1,1),(-1,1,-1),(-1,-1,-1)]),
       ((0,1,0),[(-1,1,-1),(-1,1,1),(1,1,1),(1,1,-1)]),((0,-1,0),[(-1,-1,1),(-1,-1,-1),(1,-1,-1),(1,-1,1)]),
       ((0,0,1),[(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]),((0,0,-1),[(1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1)])]
    for n,vs in F:
        b=len(P)
        for v in vs:P.append((v[0]*sx,v[1]*sy,v[2]*sz));N.append(n)
        I+=[b,b+1,b+2,b,b+2,b+3]
    return fin(P,N,I,c,r)
def cyl(r,h,c,axis='y',seg=20,r2=None):
    r2=r if r2 is None else r2;P=[];N=[];I=[]
    for i in range(seg+1):
        a=2*PI*i/seg;co,si=np.cos(a),np.sin(a)
        P+=[(r*co,-h/2,r*si),(r2*co,h/2,r2*si)];N+=[(co,0,si)]*2
    for i in range(seg):k=2*i;I+=[k,k+1,k+2,k+1,k+3,k+2]
    for sg,rad in((-1,r),(1,r2)):
        b=len(P);P.append((0,sg*h/2,0));N.append((0,sg,0))
        for i in range(seg+1):a=2*PI*i/seg;P.append((rad*np.cos(a),sg*h/2,rad*np.sin(a)));N.append((0,sg,0))
        for i in range(seg):I+=[b,b+1+i,b+2+i]
    return fin(P,N,I,c,{'y':(0,0,0),'z':(PI/2,0,0),'x':(0,0,-PI/2)}[axis])
def sph(r,c,sc=(1,1,1),seg=18,rings=12):
    P=[];N=[];I=[]
    for j in range(rings+1):
        t=PI*j/rings
        for i in range(seg+1):
            a=2*PI*i/seg;u=np.array([np.sin(t)*np.cos(a),np.cos(t),np.sin(t)*np.sin(a)])
            P.append(u*r*np.array(sc));N.append(u)
    for j in range(rings):
        for i in range(seg):
            a=j*(seg+1)+i;b=a+seg+1;I+=[a,b,a+1,a+1,b,b+1]
    return fin(P,N,I,c,(0,0,0))

MATS={}
def M(n,hexc,metal=.2,rough=.6,emis=None):
    MATS[n]=dict(name=n,pbrMetallicRoughness=dict(baseColorFactor=[(hexc>>16&255)/255,(hexc>>8&255)/255,(hexc&255)/255,1],metallicFactor=metal,roughnessFactor=rough),**({'emissiveFactor':list(emis)} if emis else {}))
class Nd:
    def __init__(s,name,t=None):s.name=name;s.t=t;s.p={}
    def add(s,mat,g):
        d=s.p.setdefault(mat,[[],[],[]]);off=sum(len(x) for x in d[0])
        d[0].append(g[0]);d[1].append(g[1]);d[2].append(g[2]+off)

def write(fn,nodes):
    bn=bytearray();views=[];acc=[];meshes=[];gn=[];mids={};mats=[]
    def buf(a,tg):
        while len(bn)%4:bn.append(0)
        raw=a.tobytes();views.append(dict(buffer=0,byteOffset=len(bn),byteLength=len(raw),target=tg));bn.extend(raw);return len(views)-1
    for nd in nodes:
        e={'name':nd.name}
        if nd.t:e['translation']=list(nd.t)
        if nd.p:
            prims=[]
            for m,(P,Nn,I) in nd.p.items():
                P=np.concatenate(P);Nn=np.concatenate(Nn);I=np.concatenate(I).astype(np.uint32)
                if m not in mids:mids[m]=len(mats);mats.append(MATS[m])
                acc.append(dict(bufferView=buf(P,34962),componentType=5126,count=len(P),type='VEC3',min=P.min(0).tolist(),max=P.max(0).tolist()))
                acc.append(dict(bufferView=buf(Nn,34962),componentType=5126,count=len(Nn),type='VEC3'))
                acc.append(dict(bufferView=buf(I,34963),componentType=5125,count=len(I),type='SCALAR'))
                prims.append(dict(attributes=dict(POSITION=len(acc)-3,NORMAL=len(acc)-2),indices=len(acc)-1,material=mids[m]))
            meshes.append(dict(name=nd.name,primitives=prims));e['mesh']=len(meshes)-1
        gn.append(e)
    while len(bn)%4:bn.append(0)
    js=json.dumps(dict(asset=dict(version='2.0',generator='make_models.py'),scene=0,scenes=[dict(nodes=list(range(len(gn))))],nodes=gn,meshes=meshes,materials=mats,accessors=acc,bufferViews=views,buffers=[dict(byteLength=len(bn))]),separators=(',',':')).encode()
    while len(js)%4:js+=b' '
    out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bn))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(bn),0x004E4942)+bytes(bn)
    open(os.path.join(OUT,fn),'wb').write(out);print(fn,len(out),'байт')

# ---------- материалы ----------
M('suit',0x2b3440,.1,.8);M('boots',0x12161c,.1,.7);M('armor',0xff4655,.25,.5);M('trim',0x1d232b,.3,.5)
M('glow',0x00ffd5,0,.4,(0,1,.83));M('visor',0xff4655,0,.3,(1,.2,.25));M('metal',0x20252c,.5,.4);M('dark',0x12161b,.3,.6)
M('glove',0x1c2229,.1,.8);M('sleeve',0x2b3440,.1,.8);M('lens',0x00e5c8,0,.2,(0,.7,.6))
M('accR',0xff4655,.3,.5);M('accT',0x00e5c8,.3,.5);M('accY',0xffd36b,.4,.4)
M('crate',0x4a6178,.35,.55);M('panel',0x5b7690,.35,.5);M('frame',0x22313f,.4,.5)
M('barrel',0x2d6a6f,.35,.5);M('hoop',0x00e5c8,0,.4,(0,.8,.7));M('lid',0x15222a,.4,.5)

# ---------- ВРАГ (смотрит в +Z, ноги на y=0, голова ~1.75 м) ----------
legs=Nd('legs');body=Nd('body');head=Nd('head');gun=Nd('gun',(.3,.98,.4))
for sx in(-1,1):
    legs.add('suit',box((.2,.5,.22),(sx*.14,.3,0)));legs.add('armor',box((.17,.1,.05),(sx*.14,.4,.12)))
    legs.add('boots',box((.22,.14,.34),(sx*.14,.07,.04)))
body.add('trim',box((.5,.16,.3),(0,.62,0)));body.add('suit',box((.58,.5,.32),(0,.95,0)))
body.add('armor',box((.52,.36,.1),(0,1.04,.16)));body.add('glow',box((.3,.04,.02),(0,1.15,.215)))
body.add('trim',box((.6,.08,.34),(0,.72,0)));body.add('suit',cyl(.08,.14,(0,1.5,0)))
for sx in(-1,1):
    body.add('armor',sph(.14,(sx*.37,1.24,0),(1,.8,1)));body.add('suit',box((.13,.4,.15),(sx*.4,1.02,.02)))
body.add('suit',box((.11,.11,.4),(.3,.88,.26)));body.add('suit',box((.11,.11,.4),(-.04,.9,.36)))
body.add('glove',box((.1,.1,.1),(.3,.88,.5)));body.add('glove',box((.1,.1,.1),(-.04,.9,.58)))
body.add('trim',box((.38,.5,.16),(0,1.02,-.24)));body.add('metal',cyl(.01,.3,(.12,1.4,-.28)))
head.add('trim',sph(.21,(0,1.76,-.01),(1,1.05,1.05)));head.add('visor',box((.28,.08,.12),(0,1.74,.16)))
head.add('glow',box((.03,.1,.1),(.21,1.75,0)));head.add('glow',box((.03,.1,.1),(-.21,1.75,0)))
head.add('armor',box((.06,.04,.3),(0,1.97,0)))
gun.add('metal',box((.07,.11,.4),(0,0,0)));gun.add('dark',cyl(.015,.35,(0,.01,.35),'z'));gun.add('metal',box((.04,.16,.08),(0,-.12,.05),(.2,0,0)))
gun.add('accR',box((.073,.02,.25),(0,.056,0)))
write('enemy.glb',[legs,body,head,gun])

# ---------- ОРУЖИЕ (ствол смотрит в -Z) ----------
def hands(n,gz,fz,sc=1):
    n.add('glove',box((.075,.08,.09),(0,-.15*sc,gz)));n.add('sleeve',box((.08,.08,.4),(0,-.22*sc,gz+.26),(.35,0,0)))
    n.add('glove',box((.085,.07,.12),(0,-.1*sc,fz)));n.add('sleeve',box((.08,.08,.4),(0,-.17*sc,fz+.05-.1),(.5,0,0)))
v=Nd('vandal');mz=Nd('muzzle',(0,.01,-1.0))
v.add('metal',box((.07,.11,.5),(0,0,0)));v.add('accR',box((.073,.02,.35),(0,.056,0)));v.add('dark',box((.06,.085,.35),(0,-.005,-.42)))
v.add('dark',cyl(.014,.35,(0,.01,-.78),'z'));v.add('dark',cyl(.024,.07,(0,.01,-.96),'z'));v.add('dark',box((.01,.04,.01),(0,.06,-.85)))
v.add('dark',box((.03,.02,.3),(0,.07,.02)));v.add('dark',box((.05,.1,.3),(0,-.01,.4)));v.add('dark',box((.05,.14,.07),(0,-.12,.12),(.25,0,0)))
v.add('metal',box((.04,.2,.09),(0,-.15,-.06),(-.2,0,0)));hands(v,.12,-.38)
write('vandal.glb',[v,mz])
c=Nd('classic');mz=Nd('muzzle',(0,.03,-.2))
c.add('metal',box((.05,.06,.26),(0,.03,-.02)));c.add('dark',box((.045,.045,.22),(0,-.02,0)));c.add('dark',cyl(.014,.06,(0,.03,-.17),'z'))
c.add('dark',box((.045,.12,.065),(0,-.09,.1),(.2,0,0)));c.add('accT',box((.052,.012,.18),(0,.062,-.02)));c.add('dark',box((.03,.01,.07),(0,-.055,-.01)))
c.add('dark',box((.01,.02,.01),(0,.07,-.13)));c.add('dark',box((.02,.02,.01),(0,.07,.1)))
c.add('glove',box((.07,.08,.08),(0,-.1,.1)));c.add('sleeve',box((.08,.08,.4),(0,-.17,.34),(.35,0,0)))
write('classic.glb',[c,mz])
o=Nd('operator');mz=Nd('muzzle',(0,.02,-1.27))
o.add('metal',box((.07,.12,.5),(0,0,0)));o.add('dark',cyl(.02,.9,(0,.02,-.7),'z'));o.add('dark',cyl(.032,.1,(0,.02,-1.2),'z'))
o.add('dark',box((.06,.13,.4),(0,-.01,.45)));o.add('dark',box((.05,.04,.2),(0,.08,.4)));o.add('accY',box((.073,.02,.3),(0,.066,.05)))
o.add('dark',cyl(.04,.4,(0,.13,-.05),'z'));o.add('metal',cyl(.056,.08,(0,.13,-.28),'z'));o.add('metal',cyl(.052,.08,(0,.13,.2),'z'))
o.add('lens',cyl(.046,.01,(0,.13,-.325),'z'));o.add('metal',box((.03,.05,.04),(0,.085,-.15)));o.add('metal',box((.03,.05,.04),(0,.085,.1)))
o.add('dark',box((.05,.14,.07),(0,-.12,.14),(.25,0,0)));o.add('metal',box((.04,.1,.1),(0,-.1,-.08)));hands(o,.14,-.4)
write('operator.glb',[o,mz])

# ---------- ЯЩИК (единичный куб, масштабируется в игре) ----------
k=Nd('crate');k.add('crate',box((1,1,1),(0,0,0)))
for a in(-.46,.46):
    for b in(-.46,.46):
        k.add('frame',box((1.02,.08,.08),(0,a,b)));k.add('frame',box((.08,1.02,.08),(a,0,b)));k.add('frame',box((.08,.08,1.02),(a,b,0)))
k.add('panel',box((.6,.6,1.01),(0,0,0)));k.add('panel',box((1.01,.6,.6),(0,0,0)));k.add('panel',box((.6,1.01,.6),(0,0,0)))
write('crate.glb',[k])

# ---------- БОЧКА (основание на y=0, высота 1.2) ----------
b=Nd('barrel');b.add('barrel',cyl(.5,1.2,(0,.6,0),'y',28));b.add('hoop',cyl(.515,.05,(0,.25,0),'y',28));b.add('hoop',cyl(.515,.05,(0,.95,0),'y',28))
b.add('frame',cyl(.508,.1,(0,.6,0),'y',28));b.add('lid',cyl(.46,.04,(0,1.2,0),'y',28));b.add('metal',cyl(.08,.05,(.2,1.23,.1)))
write('barrel.glb',[b])
