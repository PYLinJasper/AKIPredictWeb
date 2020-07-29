import psycopg2
import pandas as pd
import numpy as np
import math
import datetime

def ConvertFormat(Input): 
    creatinine=pd.DataFrame()
    intake=pd.DataFrame()
    urine=pd.DataFrame()
    pH=pd.DataFrame()
    Hct=pd.DataFrame()
    BUN=pd.DataFrame()
    Na=pd.DataFrame()
    K=pd.DataFrame()
    TP=pd.DataFrame()
    systolicBP=pd.DataFrame()
    meanBP=pd.DataFrame()
    MSI=pd.DataFrame()
    eGFR=pd.DataFrame()
    
    icustay_id=Input.groupby(Input.index//6)['icustay_id'].nth(0)
    age=Input.groupby(Input.index//6)['age'].nth(0)
    weight=Input.groupby(Input.index//6)['weight'].nth(0)
    gender=Input.groupby(Input.index//6)['gender'].nth(0)
    for i in range(0,6):
        creatinine['creatinine_seq_'+str(i)]=Input.groupby(Input.index//6)['creatinine'].nth(i)
        intake['intake_seq_'+str(i)]=Input.groupby(Input.index//6)['intake'].nth(i)
        urine['urine_seq_'+str(i)]=Input.groupby(Input.index//6)['urine'].nth(i)
        pH['pH_seq_'+str(i)]=Input.groupby(Input.index//6)['pH'].nth(i)
        Hct['Hct_seq_'+str(i)]=Input.groupby(Input.index//6)['Hct'].nth(i)
        BUN['Bun_seq_'+str(i)]=Input.groupby(Input.index//6)['BUN'].nth(i)
        Na['Na_seq_'+str(i)]=Input.groupby(Input.index//6)['Na'].nth(i)
        K['K_seq_'+str(i)]=Input.groupby(Input.index//6)['K'].nth(i)
        TP['TP_seq_'+str(i)]=Input.groupby(Input.index//6)['TP'].nth(i)
        systolicBP['systolicBP_seq_'+str(i)]=Input.groupby(Input.index//6)['systolicBP'].nth(i)
        meanBP['meanBP_seq_'+str(i)]=Input.groupby(Input.index//6)['meanBP'].nth(i)
        MSI['MSI_seq_'+str(i)]=Input.groupby(Input.index//6)['MSI'].nth(i)
        eGFR['eGFR_seq_'+str(i)]=Input.groupby(Input.index//6)['eGFR'].nth(i)
    
    xgb_input=pd.DataFrame()
    xgb_input['icustay_id']=icustay_id
    xgb_input['age']=age
    xgb_input['weight']=weight
    xgb_input['gender']=gender
    xgb_input=pd.concat([xgb_input,creatinine,intake,urine,pH,Hct,BUN,Na,K,TP,systolicBP,meanBP,MSI,eGFR],axis=1)
    xgb_input['aki']=Input.groupby(Input.index//6)['aki'].nth(0)
    xgb_input=xgb_input.set_index('icustay_id')
    return xgb_input

def GetIntakeRes(sqlstr,sqlstr_Time):
    intake_res=pd.DataFrame()
    cur.execute("SELECT TB.icustay_id,IC.charttime,(CASE WHEN IC.amountuom = 'tsp' THEN IC.amount*5 WHEN IC.amountuom = 'mg' THEN IC.amount/1000 WHEN IC.amountuom = 'mcg' THEN IC.amount/1000000 ELSE IC.amount END) AS amount FROM ("+sqlstr+") TB,mimiciii.inputevents_cv IC WHERE IC.amountuom IN ('ml','tsp','cc','gm','mg','mcg') AND TB.subject_id=IC.subject_id AND IC.amount IS NOT NULL AND IC.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND IC.charttime < TB.charttime ORDER BY icustay_id,charttime")
    intakeCV_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','seq_res'])
    intakeCV_res=intakeCV_res.groupby(['icustay_id','charttime']).agg({'seq_res':np.sum}).reset_index()
    
    cur.execute("SELECT TB.icustay_id,IM.starttime,(CASE WHEN IM.endtime>TB.charttime THEN TB.charttime ELSE IM.endtime END) AS endtime,(CASE WHEN IM.amountuom='L' THEN IM.amount*1000 WHEN IM.amountuom='ounces' THEN IM.amount*29.574 WHEN IM.amountuom='uL' THEN IM.amount/1000000 WHEN IM.amountuom='mcg' THEN IM.amount*29.574 ELSE IM.amount END),IM.amountuom,IM.rate,IM.rateuom FROM ("+sqlstr+") TB,mimiciii.inputevents_mv IM WHERE IM.amountuom IN ('ml','L','ounces','grams','uL','mcg') AND IM.statusdescription!='Rewritten' AND TB.subject_id=IM.subject_id AND IM.amount IS NOT NULL AND IM.starttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND IM.starttime < TB.charttime ORDER BY icustay_id,starttime")
    intakeMV_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','endtime','amount','amountuom','rate','rateuom'])
    intakeMV_res['TimeOffset']=(intakeMV_res['endtime']-intakeMV_res['charttime']).dt.total_seconds().div(3600)
    intakeMV_res['seq_res']=np.where(intakeMV_res['rateuom']=='mL/hour',intakeMV_res['rate']*intakeMV_res['TimeOffset'],intakeMV_res['amount'])
    intakeMV_res['seq_res']=np.where(intakeMV_res['endtime']==intakeMV_res['charttime'],0,intakeMV_res['seq_res'])
    intakeMV_res=intakeMV_res.drop(columns=['endtime','amount','amountuom','rate','rateuom','TimeOffset'],axis=1)
    intakeMV_res=intakeMV_res.groupby(['icustay_id','charttime']).agg({'seq_res':np.sum}).reset_index()
    
    intake_res=pd.concat([intakeCV_res,intakeMV_res],axis=0,join='inner')
    intake_res=intake_res.sort_values(by=['icustay_id','charttime']).reset_index(drop=True)
    return intake_res


def HandleMissingVal(feature_name,feature_seqres,new_df,fill_type):
    res = pd.DataFrame()
    res = pd.merge(new_df,feature_seqres,on='icustay_id',how='left')
    res['value']=np.where((res['charttime_y']-res['charttime_x']>=datetime.timedelta(hours=0))&(res['charttime_y']-res['charttime_x']<datetime.timedelta(hours=4)),res['seq_res'],np.nan)
    
    if(feature_name == 'urine' or feature_name == 'intake' or feature_name == 'TV'):
        res=res.groupby(['icustay_id','charttime_x']).agg({'value':np.sum}).reset_index()
        res['value']=np.where(res.groupby(['icustay_id'])['value'].transform('sum')==0,np.nan,res['value'])
    else:
        if(fill_type == 'Copy prev or next'):
            res=res.groupby(['icustay_id','charttime_x'])['value'].agg({np.max,'last','first'}).reset_index().rename(columns={"amax":"value"})
            res['last']=res.groupby(['icustay_id'])['last'].ffill()
            res['first']=res.groupby(['icustay_id'])['first'].bfill()
            res['value']=np.where(res['value'].isnull()&res['first'].notnull(),res['first'],res['value'])
            res['value']=np.where(res['value'].isnull()&res['last'].notnull(),res['last'],res['value'])
            res=res.drop(columns=['first','last'],axis=1)  
        if(fill_type == 'Mean'):
            res=res.groupby(['icustay_id','charttime_x']).agg({'value':np.max}).reset_index()
            res['value'].fillna(value=res.groupby('icustay_id')['value'].transform('mean'), inplace=True)
        if(fill_type == 'Zero'):
            res=res.groupby(['icustay_id','charttime_x']).agg({'value':np.max}).reset_index()
            # res['value'].fillna(value=0, inplace=True)
            # res['value']=np.where(res.groupby(['icustay_id'])['value'].transform('sum')==0,np.nan,res['value'])
        
    res=res.drop(columns=['charttime_x','icustay_id'],axis=1).rename(columns={"value":feature_name})
    return res    


def GenerateRes(sqlstr,sqlstr_Time,originInICU,fill_type):
    cur = conn.cursor()
    
    new=pd.concat([originInICU]*int(int(sqlstr_Time)/4), ignore_index=True)
    new=new.sort_values(by=['icustay_id']).reset_index(drop=True)
    new['cumCount']=new.groupby(['icustay_id']).cumcount()
    new['new_charttime']=new['charttime']-(new['cumCount']+1)*datetime.timedelta(hours=4)
    new=new.sort_values(by=['icustay_id','new_charttime']).reset_index(drop=True).drop(columns=['charttime','cumCount'],axis=1).rename(columns={"new_charttime":"charttime"})
    
    ###########creatinine##########
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid=50912 AND labevents.valuenum >0.1 AND labevents.valuenum <20 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    creatinine_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    creatinine_res=creatinine_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    creatinine_seqres=creatinine_res.rename(columns={"valuenum":"seq_res"})
    creatinine=HandleMissingVal('creatinine',creatinine_seqres,new,fill_type)


    print('creatinine')
    
    #############intake############
    intake_seqres=GetIntakeRes(sqlstr,sqlstr_Time)
    intake=HandleMissingVal('intake',intake_seqres,new,fill_type)
      
    print('intake')
    
    
    #############urine#############
    cur.execute("SELECT TB.icustay_id,outputevents.charttime,outputevents.value FROM ("+sqlstr+") TB,mimiciii.outputevents outputevents WHERE outputevents.itemid IN (40055,226559,40069) AND TB.subject_id=outputevents.subject_id AND outputevents.value IS NOT NULL AND outputevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND outputevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    urine_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','value'])
    urine_res=urine_res.groupby(['icustay_id','charttime']).agg({'value':np.min}).reset_index()

    #calculate delta  
    urine_res['delta_time']=urine_res.groupby(['icustay_id'])['charttime'].diff().dt.total_seconds().div(60)
    urine_res['rate']=np.where(urine_res['delta_time'].notnull(),urine_res['value']/urine_res['delta_time'],-1)
    urine_res=urine_res[urine_res['rate'] < (600/60)].reset_index(drop=True).drop(columns=['delta_time','rate'])
    urine_seqres=urine_res.rename(columns={"value":"seq_res"})
    
    urine=HandleMissingVal('urine',urine_seqres,new,fill_type)
    
    print('urine')
    
    #############pH###############
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid=50820 AND labevents.valuenum BETWEEN 6.8 AND 7.8 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    pH_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    pH_res=pH_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    pH_seqres=pH_res.rename(columns={"valuenum":"seq_res"})
    pH=HandleMissingVal('pH',pH_seqres,new,fill_type)

    
    print('pH')
    
    ############Hct################
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid IN (50810,51221) AND labevents.valuenum BETWEEN 10 AND 70 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    Hct_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    Hct_res=Hct_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    Hct_seqres=Hct_res.rename(columns={"valuenum":"seq_res"})
    Hct=HandleMissingVal('Hct',Hct_seqres,new,fill_type)
    
    
    print('Hct')
    
    ###########Tidal volume##########
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid=50826 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    TV_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    TV_res=TV_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    TV_seqres=TV_res.rename(columns={"valuenum":"seq_res"})
    TV=HandleMissingVal('TV',TV_seqres,new,fill_type)
        
    print('TV')
    
    ##########urea nitrogen###########
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid=51006 AND labevents.valuenum BETWEEN 0 AND 60 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    BUN_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    BUN_res=BUN_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    BUN_seqres=BUN_res.rename(columns={"valuenum":"seq_res"})
    BUN=HandleMissingVal('BUN',BUN_seqres,new,fill_type)    

    print('BUN')
    
    ##########Sodium(Na)###############
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid IN (50824,50983) AND labevents.valuenum BETWEEN 100 AND 180 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    Na_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    Na_res=Na_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    Na_seqres=Na_res.rename(columns={"valuenum":"seq_res"})
    Na=HandleMissingVal('Na',Na_seqres,new,fill_type)
    
    
    print('Sodium')
    
    ##########Potassium(K)##############
    cur.execute("SELECT TB.icustay_id,labevents.charttime,labevents.valuenum FROM ("+sqlstr+") TB,mimiciii.labevents labevents WHERE labevents.itemid IN (50822,50971) AND labevents.valuenum BETWEEN 1 AND 9 AND TB.subject_id=labevents.subject_id AND labevents.valuenum IS NOT NULL AND labevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND labevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    K_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    K_res=K_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    K_seqres=K_res.rename(columns={"valuenum":"seq_res"})
    K=HandleMissingVal('K',K_seqres,new,fill_type)    

    
    print('potassium')
    
    ##########day in ICU#############
    cur.execute("SELECT TB.icustay_id,TB.charttime,icustays.intime FROM ("+sqlstr+") TB,mimiciii.icustays icustays WHERE TB.icustay_id=icustays.icustay_id")
    day_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','intime'])
    day_res['day']=(day_res['charttime']-day_res['intime']).dt.total_seconds().div(86400)
    day_res=day_res.drop(columns=['charttime','intime'])
    day=pd.merge(new,day_res,on='icustay_id',how='left').drop(columns=['subject_id','icustay_id','charttime'])
    
    
    ##########weight#################
    cur.execute("SELECT TB.icustay_id,WT.weight FROM ("+sqlstr+") TB,weightfirstday WT WHERE TB.icustay_id=WT.icustay_id AND WT.weight >=10 AND WT.weight <200")
    weight_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','weight'])
    weight=pd.merge(new,weight_res,on='icustay_id',how='left').drop(columns=['subject_id','icustay_id','charttime'])    
    
    ##########height#################
    # cur.execute("SELECT TB.icustay_id,HT.height FROM ("+sqlstr+") TB,mimiciii.heightfirstday HT WHERE TB.icustay_id=HT.icustay_id")
    # height_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','height'])
#    height=pd.merge(new,height_res,on='icustay_id',how='left').drop(columns=['subject_id'])
    
    ###########age#################
    cur.execute("SELECT TB.icustay_id,ID.age FROM ("+sqlstr+") TB,mimiciii.icustay_detail ID WHERE TB.icustay_id=ID.icustay_id")
    age_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','age'])
    age_res.loc[age_res['age']>300,'age']=90
    age=pd.merge(new,age_res,on='icustay_id',how='left').drop(columns=['subject_id','charttime'])
    
    #########ethnicity#############
    cur.execute("SELECT TB.icustay_id,ID.ethnicity_grouped FROM ("+sqlstr+") TB,mimiciii.icustay_detail ID WHERE TB.icustay_id=ID.icustay_id")
    ethnicity_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','ethnicity'])
    
    
    ##########gender###############
    cur.execute("SELECT TB.icustay_id,PT.gender FROM ("+sqlstr+") TB,mimiciii.patients PT WHERE TB.subject_id=PT.subject_id")
    gender_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','gender'])
    gender_res['gender']=gender_res['gender'].replace({'M':1,'F':0})
    gender=pd.merge(new,gender_res,on='icustay_id',how='left').drop(columns=['subject_id','icustay_id','charttime'])
    
    #########temperature#############
    cur.execute("SELECT TB.icustay_id,chartevents.charttime,chartevents.valuenum FROM ("+sqlstr+") TB,mimiciii.chartevents chartevents WHERE chartevents.itemid IN (676,677) AND chartevents.valuenum BETWEEN 35 AND 42 AND TB.subject_id=chartevents.subject_id AND chartevents.valuenum IS NOT NULL AND chartevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND chartevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    TP_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    TP_res=TP_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    TP_seqres=TP_res.rename(columns={"valuenum":"seq_res"})
    TP=HandleMissingVal('TP',TP_seqres,new,fill_type)
    
  
    print('TP')
    
    ##########Non Invasive systolicBP#############
    cur.execute("SELECT TB.icustay_id,chartevents.charttime,chartevents.valuenum FROM ("+sqlstr+") TB,mimiciii.chartevents chartevents WHERE chartevents.itemid=220179  AND chartevents.valuenum<250 AND TB.subject_id=chartevents.subject_id AND chartevents.valuenum IS NOT NULL AND chartevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND chartevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    systolicBP_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    systolicBP_res=systolicBP_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    systolicBP_seqres=systolicBP_res.rename(columns={"valuenum":"seq_res"})
    systolicBP=HandleMissingVal('systolicBP',systolicBP_seqres,new,fill_type)
    
    print('systolic BP')
    

    
    #########Non Invasive Blood Pressure mean##########
    cur.execute("SELECT TB.icustay_id,chartevents.charttime,chartevents.valuenum FROM ("+sqlstr+") TB,mimiciii.chartevents chartevents WHERE chartevents.itemid=220181 AND chartevents.valuenum<200 AND TB.subject_id=chartevents.subject_id AND chartevents.valuenum IS NOT NULL AND chartevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND chartevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    meanBP_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    meanBP_res=meanBP_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    meanBP_seqres=meanBP_res.rename(columns={"valuenum":"seq_res"})
    meanBP=HandleMissingVal('meanBP',meanBP_seqres,new,fill_type)   
    
    print('mean BP')
    
    #########Non Invasive mean arterial blood############
    cur.execute("SELECT TB.icustay_id,chartevents.charttime,chartevents.valuenum FROM ("+sqlstr+") TB,mimiciii.chartevents chartevents WHERE chartevents.itemid=220052 AND chartevents.valuenum BETWEEN 1 AND 200 AND TB.subject_id=chartevents.subject_id AND chartevents.valuenum IS NOT NULL AND chartevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND chartevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    arterialBP_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    arterialBP_res=arterialBP_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    arterialBP_seqres=arterialBP_res.rename(columns={"valuenum":"seq_res"})
    



    
    ##############MSI#################
    cur.execute("SELECT TB.icustay_id,chartevents.charttime,chartevents.valuenum FROM ("+sqlstr+") TB,mimiciii.chartevents chartevents WHERE chartevents.itemid=220045 AND chartevents.valuenum BETWEEN 0 AND 300 AND TB.subject_id=chartevents.subject_id AND chartevents.valuenum IS NOT NULL AND chartevents.charttime >= (TB.charttime - interval '"+sqlstr_Time+"' hour) AND chartevents.charttime < TB.charttime ORDER BY icustay_id,charttime")
    HR_res=pd.DataFrame(cur.fetchall(),columns=['icustay_id','charttime','valuenum'])
    HR_res=HR_res.groupby(['icustay_id','charttime']).agg({'valuenum':np.min}).reset_index()
    HR_seqres=HR_res.rename(columns={"valuenum":"seq_res"})    
    MSI_seqres=pd.merge(HR_seqres,arterialBP_seqres,on=['icustay_id','charttime'],how='inner')
    MSI_seqres['seq_res']=MSI_seqres['seq_res_x']/MSI_seqres['seq_res_y']
    MSI_seqres=MSI_seqres.groupby(['icustay_id','charttime']).agg({'seq_res':np.min}).reset_index()
    MSI=HandleMissingVal('MSI',MSI_seqres,new,fill_type)
    
    
    print('MSI')
    
    #############eGFR##################
    eGFR_res=pd.DataFrame()
    eGFR_res=pd.merge(age_res,ethnicity_res,on='icustay_id',how='left')
    eGFR_res=pd.merge(eGFR_res,gender_res,on='icustay_id',how='left')
    # eGFR_res=pd.merge(eGFR_res,height_res,on='icustay_id',how='left')
    eGFR_res=pd.merge(creatinine_seqres,eGFR_res,on='icustay_id',how='left')
    
    #calculate eGFR
    eGFR = []

    for i in range(len(eGFR_res)):
        if(eGFR_res['age'][i]<18 or eGFR_res['age'][i]>200):
            eGFR.append(np.nan)
        else:
            if(eGFR_res['ethnicity'][i]=='black' and eGFR_res['gender'][i]==0):
                if(eGFR_res['seq_res'][i] <= 0.7):
                    eGFR.append(166*(eGFR_res['seq_res'][i]/0.7)**(-0.329)*0.993**float(eGFR_res['age'][i]))
                else:
                    eGFR.append(166*(eGFR_res['seq_res'][i]/0.7)**(-1.209)*0.993**float(eGFR_res['age'][i]))
            elif(eGFR_res['ethnicity'][i]=='black' and eGFR_res['gender'][i]==1):
                if(eGFR_res['seq_res'][i] <= 0.9):
                    eGFR.append(163*(eGFR_res['seq_res'][i]/0.9)**(-0.411)*0.993**float(eGFR_res['age'][i]))
                else:
                    eGFR.append(163*(eGFR_res['seq_res'][i]/0.9)**(-1.209)*0.993**float(eGFR_res['age'][i]))                        
            elif(eGFR_res['ethnicity'][i]!='black' and eGFR_res['gender'][i]==0):
                if(eGFR_res['seq_res'][i] <= 0.7):
                    eGFR.append(144*(eGFR_res['seq_res'][i]/0.7)**(-0.329)*0.993**float(eGFR_res['age'][i]))
                else:
                    eGFR.append(144*(eGFR_res['seq_res'][i]/0.7)**(-1.209)*0.993**float(eGFR_res['age'][i])) 
            elif(eGFR_res['ethnicity'][i]!='black' and eGFR_res['gender'][i]==1):
                if(eGFR_res['seq_res'][i] <= 0.9):
                    eGFR.append(141*(eGFR_res['seq_res'][i]/0.9)**(-0.411)*0.993**float(eGFR_res['age'][i]))
                else:
                    eGFR.append(141*(eGFR_res['seq_res'][i]/0.9)**(-1.209)*0.993**float(eGFR_res['age'][i]))    
            else:
                eGFR.append(np.nan)

    eGFR_res['eGFR'] = eGFR
    eGFR.index(max(eGFR_res['eGFR']))

    
    eGFR_res=eGFR_res.drop(['seq_res','age','ethnicity','gender'],axis=1)
    eGFR_seqres=eGFR_res.rename(columns={"eGFR":"seq_res"})
    eGFR=HandleMissingVal('eGFR',eGFR_seqres,new,fill_type)    

    
    print('eGFR')
    

    res=pd.DataFrame()
    res=pd.concat([age,day,weight,gender,creatinine,intake,urine,pH,Hct,BUN,Na,K,TP,systolicBP,meanBP,MSI,eGFR],axis=1)
    res.set_index('icustay_id',inplace=True)
    
    return res    


if __name__== "__main__":
    conn = psycopg2.connect(host="",database=""
    , user="", password="",port="")
    cur = conn.cursor()


    cur.execute("SELECT icustays.subject_id,NAP.icustay_id,NAP.charttime FROM noaki NAP,mimiciii.icustays icustays WHERE NAP.icustay_id=icustays.icustay_id AND subject_id NOT IN (SELECT subject_id FROM mimiciii.procedures_icd WHERE icd9_code IN ('5498','3995')) ORDER BY icustay_id")
    kdigoNoAKIfilterDialysis_res = pd.DataFrame(cur.fetchall(),columns=['subject_id','icustay_id','charttime'])
    
    cur.execute("SELECT icustays.subject_id,TB.icustay_id,TB.charttime FROM (SELECT MIN(KS.icustay_id) AS icustay_id,MIN(KS.charttime) AS charttime,MIN(KS.aki_stage) AS aki_stage FROM kdigo_stages KS,mimiciii.icustays icustays WHERE KS.icustay_id=icustays.icustay_id AND aki_stage!=0 AND KS.charttime BETWEEN icustays.intime AND icustays.outtime GROUP BY KS.icustay_id ORDER BY KS.icustay_id) TB,mimiciii.icustays icustays WHERE TB.icustay_id=icustays.icustay_id AND subject_id NOT IN (SELECT subject_id FROM mimiciii.procedures_icd WHERE icd9_code IN ('5498','3995'))")
    kdigoAKIfilterDialysis_res = pd.DataFrame(cur.fetchall(),columns=['subject_id','icustay_id','charttime'])

    AKIsqlstr="SELECT icustays.subject_id,TB.icustay_id,TB.charttime FROM (SELECT MIN(KS.icustay_id) AS icustay_id,MIN(KS.charttime) AS charttime,MIN(KS.aki_stage) AS aki_stage FROM kdigo_stages KS,mimiciii.icustays icustays WHERE KS.icustay_id=icustays.icustay_id AND aki_stage!=0 AND KS.charttime BETWEEN icustays.intime AND icustays.outtime GROUP BY KS.icustay_id ORDER BY KS.icustay_id) TB,mimiciii.icustays icustays WHERE TB.icustay_id=icustays.icustay_id AND subject_id NOT IN (SELECT subject_id FROM mimiciii.procedures_icd WHERE icd9_code IN ('5498','3995'))"
    NoAKIsqlstr="SELECT icustays.subject_id,NAP.icustay_id,NAP.charttime FROM noaki NAP,mimiciii.icustays icustays WHERE NAP.icustay_id=icustays.icustay_id AND subject_id NOT IN (SELECT subject_id FROM mimiciii.procedures_icd WHERE icd9_code IN ('5498','3995')) ORDER BY icustay_id"
    
    TimeOffset="24"                      #How many hours data you want to get
    FillType="Copy prev or next"         #Can be 'Copy prev or next' or 'Mean' or 'Zero'


    AKIPT=GenerateRes(AKIsqlstr,TimeOffset,kdigoAKIfilterDialysis_res,FillType)
    AKIPT['aki'] = 1  
    
    print(AKIPT)
    
    NoAKIPT=GenerateRes(NoAKIsqlstr,TimeOffset,kdigoNoAKIfilterDialysis_res,FillType)
    NoAKIPT['aki'] = 0
    
    print(NoAKIPT)
    
    TrainPT = pd.concat([AKIPT,NoAKIPT],axis=0,join='inner')
    TrainPT=TrainPT.loc[TrainPT['age']>18]
    TrainPT.to_csv('MIMIC_LSTM_Input.csv', encoding='utf-8')
    
    LSTM_Input=pd.read_csv('MIMIC_LSTM_Input.csv')
    XgbFile=ConvertFormat(LSTM_Input)
    XgbFile.to_csv('MIMIC_Xgboost_Input.csv', encoding='utf-8')
    
