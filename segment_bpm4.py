"""Сегментация B2C-клиентов для ДЗ 3 (команда T12, датасет Ритейл БПМ4).
Вход: исходные CSV пакета (Receipts, Receipt_Lines, Customers, Products, Categories).
Клиент: основная карта (дубли объединены по master_customer_id); чеки без тестовых, сотрудников и партнеров.
RFM: окно 12 месяцев до даты состояния; границы из metadata/metrics.csv пакета:
  R3 <= 30 дней, R2 31-90, R1 > 90; F3 >= 6 чеков, F2 2-5, F1 = 1; M3 >= 12000 руб, M2 3000-11999, M1 < 3000.
Новые: первая покупка по CRM в последние 90 дней до даты состояния.
Кластеры: K-means (k=5, random_state=42) на снимке 31.12.2025; признаки: log(1+давность), log(1+число чеков),
  log(средний чек), число групп категорий, доли выручки по 9 группам (вес 1/3); z-нормировка по снимку 31.12.2025.
  Центры, средние и стандартные отклонения сохраняются и применяются к снимкам других месяцев.
"""
import sys, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
D=sys.argv[1] if len(sys.argv)>1 else '/home/claude/in/z/Retail_BPM4/data/'
OUT=sys.argv[2] if len(sys.argv)>2 else '/home/claude/seg/out/'
rd=lambda f: pd.read_csv(D+f,sep=';',encoding='utf-8-sig',dtype=str)
R=rd('Receipts.csv'); C=rd('Customers.csv'); L=rd('Receipt_Lines.csv'); P=rd('Products.csv'); K=rd('Categories.csv')
R=R[(R.is_test=='false')&(R.is_employee=='false')&(R.is_b2b_or_partner=='false')&R.customer_id.notna()].copy()
master=dict(zip(C.customer_id, C.master_customer_id.fillna(C.customer_id)))
R['client']=R.customer_id.map(master); R['date']=pd.to_datetime(R.receipt_datetime.str[:10]); R['net']=R.net_amount.astype(float)
crm=pd.to_datetime(C[C.customer_key_status=='Основная карта'].set_index('customer_id').crm_first_purchase_date)
L=L[L.receipt_id.isin(R.receipt_id)].copy(); L['net']=L.net_amount.astype(float)
L=L.merge(P.merge(K,on='category_id')[['sku_id','category_l1']],on='sku_id').merge(R[['receipt_id','client','date']],on='receipt_id')
GROUPS=sorted(L.category_l1.unique())
def win(d):
    d=pd.Timestamp(d); return d-pd.DateOffset(months=12)+pd.Timedelta(days=1), d
def features(d):
    s,d=win(d); r=R[(R.date>=s)&(R.date<=d)]
    g=r.groupby('client').agg(last=('date','max'),F=('receipt_id','nunique'),M=('net','sum'))
    g['R']=(d-g['last']).dt.days
    l=L[(L.date>=s)&(L.date<=d)]
    sh=l.pivot_table(index='client',columns='category_l1',values='net',aggfunc='sum',fill_value=0).reindex(columns=GROUPS,fill_value=0)
    sh=sh.div(sh.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
    g['ncat']=l.groupby('client').category_l1.nunique().reindex(g.index).fillna(0)
    g['avgcheck']=g.M/g.F
    return g.join(sh.add_prefix('sh_'))
def macro(rs,fs,new):
    if new: return 'Новые'
    if rs>=2 and fs==3: return 'Ценные активные'
    if rs>=2 and fs==2: return 'Регулярные'
    if rs>=2: return 'Разовые'
    if fs>=2: return 'Под угрозой ухода'
    return 'Спящие'
def snapshot(d):
    g=features(d); dd=pd.Timestamp(d)
    g['Rs']=np.where(g.R<=30,3,np.where(g.R<=90,2,1)); g['Fs']=np.where(g.F>=6,3,np.where(g.F>=2,2,1)); g['Ms']=np.where(g.M>=12000,3,np.where(g.M>=3000,2,1))
    first=crm.reindex(g.index); g['new']=(first>dd-pd.Timedelta(days=90))&(first<=dd)
    g['code']=['R%dF%dM%d'%x for x in zip(g.Rs,g.Fs,g.Ms)]
    g['group']=[macro(a,b,n) for a,b,n in zip(g.Rs,g.Fs,g.new)]
    return g
FC=['R','F','avgcheck','ncat']+['sh_'+x for x in GROUPS]
W=pd.Series(1.0,index=FC); W[['sh_'+x for x in GROUPS]]=1/3
def transform(g):
    X=g[FC].copy(); X['R']=np.log1p(X.R); X['F']=np.log1p(X.F); X['avgcheck']=np.log(X.avgcheck); return X
base=snapshot('2025-12-31'); X=transform(base); MU,SD=X.mean(),X.std(); Z=(X-MU)/SD*W
km=KMeans(5,n_init=20,random_state=42).fit(Z); base['cl']=km.labels_
sil=silhouette_score(Z,km.labels_,sample_size=6000,random_state=1)
prof=base.groupby('cl').agg(R=('R','median'),F=('F','median'),chk=('avgcheck','median'),ncat=('ncat','median'),perf=('sh_Парфюмерия','mean'),hh=('sh_Бытовая химия','mean'))
names={}
names[prof.F.idxmax()]='1 Частые, широкая корзина'
rest=prof.drop(list(names))
names[rest.perf.idxmax()]='3 Редкие крупные, парфюмерия'
rest=prof.drop(list(names)); names[rest.F.idxmax()]='2 Регулярные, средний ритм'
rest=prof.drop(list(names)); names[rest.chk.idxmin()]='5 Разовые мелкие, быт и гигиена'
rest=prof.drop(list(names)); names[rest.index[0]]='4 Редкие, уход и декор'
CENT=pd.DataFrame(km.cluster_centers_,columns=FC)
def assign(g):
    Zg=(transform(g)-MU)/SD*W
    dist=((Zg.values[:,None,:]-CENT.values[None,:,:])**2).sum(axis=2)
    return [names[i] for i in dist.argmin(axis=1)]
base['cluster']=assign(base); assert (base['cluster']==base['cl'].map(names)).all()
import os; os.makedirs(OUT,exist_ok=True)
# snapshots: month-ends Dec 2024 .. Dec 2025
ends=pd.date_range('2024-12-31','2025-12-31',freq='ME')
snaps=[]
for d in ends:
    g=snapshot(d); g['cluster']=assign(g); g['Дата состояния']=d.date(); g['Месяц']=d.strftime('%Y-%m'); snaps.append(g)
S=pd.concat(snaps).reset_index()
S2=S.rename(columns={'client':'client_key','R':'R, дней','F':'F, чеков','M':'M, руб','code':'RFM-код','group':'RFM-группа','cluster':'Кластер'})
S2[['Дата состояния','Месяц','client_key','R, дней','F, чеков','M, руб','Rs','Fs','Ms','RFM-код','RFM-группа','Кластер']].to_csv(OUT+'rfm_snapshots.csv',sep=';',index=False,encoding='utf-8-sig',float_format='%.2f')
# clients
allc=pd.Index(sorted(R.client.unique()),name='client')
s24=snapshot('2024-12-31'); e25=base
cl=pd.DataFrame(index=allc)
cl['first']=crm.reindex(allc).dt.date
cl['RFM-код']=e25.code.reindex(allc); cl['RFM-группа']=e25.group.reindex(allc).fillna('Без покупок в окне')
cl['R, дней']=e25.R.reindex(allc); cl['F, чеков']=e25.F.reindex(allc); cl['M, руб']=e25.M.reindex(allc)
cl['Средний чек в окне, руб']=e25.avgcheck.reindex(allc); cl['Групп категорий']=e25.ncat.reindex(allc)
cl['Кластер']=e25.cluster.reindex(allc).fillna('Без покупок в окне')
start=s24.group.reindex(allc)
fp=pd.to_datetime(cl['first'])
cl['Группа 31.12.2024']=np.where(start.notna(),start,np.where(fp>pd.Timestamp('2024-12-31'),'Новые после 31.12.2024','Без покупок в окне'))
cl['Группа 31.12.2025']=cl['RFM-группа']
cl=cl.reset_index().rename(columns={'client':'client_key','first':'Первая покупка (CRM)'})
cl.to_csv(OUT+'clients_segments.csv',sep=';',index=False,encoding='utf-8-sig',float_format='%.2f')
# centers and scaling
cen=pd.DataFrame(km.cluster_centers_,columns=FC); cen.insert(0,'Кластер',[names[i] for i in range(5)])
cen['Тип строки']='центр (z, с весами)'
mu=MU.to_frame().T; mu.insert(0,'Кластер','-'); mu['Тип строки']='среднее (до z)'
sd=SD.to_frame().T; sd.insert(0,'Кластер','-'); sd['Тип строки']='ст. отклонение (до z)'
w=W.to_frame().T; w.insert(0,'Кластер','-'); w['Тип строки']='вес признака'
pd.concat([cen,mu,sd,w]).to_csv(OUT+'cluster_centers.csv',sep=';',index=False,encoding='utf-8-sig',float_format='%.6f')
print('silhouette',round(sil,3)); print(cl['RFM-группа'].value_counts()); print(cl['Кластер'].value_counts()); print(pd.crosstab(cl['Группа 31.12.2024'],cl['Группа 31.12.2025']))
print(S.groupby(['Месяц']).size().to_string())
