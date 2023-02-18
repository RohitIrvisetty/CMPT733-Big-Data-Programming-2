# anomaly_detection.py
import pandas as pd
import numpy as np
import ast
import math
from sklearn.cluster import KMeans
class AnomalyDetection():
    
    def scaleNum(self, df, indices):
        """
            Write your code!
        """
        df21=pd.DataFrame([])
        df21=df["features"]
        df21=df21.apply(lambda x:pd.DataFrame(x)[0])
        for i in range(38):
            m=df21[12+i].mean()
            sd=df21[12+i].std()
            df21[12+i]=df21[12+i].apply(lambda x:x-(m)/sd)
        df23=pd.DataFrame([])
        df23=df["features"]
        for i in range(len(df23)):
            df23[i]=list(df21.iloc[i])
            df23[i]=[0 if math.isnan(x) else x for x in df23[i]]
        df24=pd.DataFrame([])
        df24["id"]=df23.index
        df24["features"]=df23
        df24=df24.set_index('id')
        return df24

    def cat2Num(self, df, indices):
        """
            Write your code!
        """
        arr1=list()
        arr2=[0,0,0,0,0,0,0,0,0,0,0,0]
        df11=df.reset_index(drop=True)
        df11=df11["features"]
        df11=df11.apply(lambda x:ast.literal_eval(x))
        for i in range(len(df11)):
            arr1.append(df11[i][0])
            arr1.append(df11[i][1])
        arr1=np.unique(arr1)
        arr1=list(arr1)
        arr1=sorted(arr1,key=str.casefold)
        df12=df11.apply(lambda x:x[2:])
        df13=df12.apply(lambda x:arr2+x)
        for i in range(len(df13)):
            index1=arr1.index(df11[i][0])
            index2=arr1.index(df11[i][1])
            df13[i][index1]=1
            df13[i][index2]=1
        df14=pd.DataFrame([])
        df14["id"]=df13.index
        df14["features"]=df13
        df14=df14.set_index('id')
        return df14
    
    def detect(self, df, k, t):
        """
            Write your code!
        """
        df31=df["features"]
        arr_32=[]
        for i in range(len(df31)):
            arr_32.append(df31[i])
        df_model=KMeans(n_clusters=k).fit(arr_32)
        arr_35=[]
        arr_35.append(df_model.labels_)
        df36=pd.DataFrame([])
        max32=max(arr_35[0])
        min35=min(arr_35[0])
        den=max32-min35
        arr_36=[]
        #arr_35[0]=arr_35[0].apply(lambda x:max32-x)
        for i in range(len(arr_35)):
            arr_36.append(max32-arr_35[0][i]/den)
        print(max32,min35)
        print(arr_36[0])
        df36["score"]=arr_36[0]
        df36=df36[df36["score"]>=0.97]
        print(df36)
        df["score"]=df36
        return df


if __name__ == "__main__":
    df = pd.read_csv('A5-data/logs-features-sample.csv').set_index('id')
    ad = AnomalyDetection()
    a1=[]
    df1 = ad.cat2Num(df, [0,1])
    print(df1)

    df2 = ad.scaleNum(df1, [6])
    print(df2)

    df3 = ad.detect(df2, 8, 0.97)
    print(df3)
    print(len(df3))