import json
from django.shortcuts import render
from django.core.paginator import Paginator
from django.db.models import Q
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import User
from django.core.files.storage import FileSystemStorage
from django.contrib import messages

from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.http import HttpResponseRedirect
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from django.contrib import auth
import xgboost as xgb
from xgboost import XGBClassifier
import os
import numpy as np
import pickle



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def ConvertFormat(Input):
    creatinine = pd.DataFrame()
    intake = pd.DataFrame()
    urine = pd.DataFrame()
    pH = pd.DataFrame()
    Hct = pd.DataFrame()
    BUN = pd.DataFrame()
    Na = pd.DataFrame()
    K = pd.DataFrame()
    TP = pd.DataFrame()
    systolicBP = pd.DataFrame()
    meanBP = pd.DataFrame()
    MSI = pd.DataFrame()
    eGFR = pd.DataFrame()

    age = Input.groupby(Input.index // 6)['age'].nth(0)
    # day=Input.groupby(Input.index//6)['day'].nth(0)
    weight = Input.groupby(Input.index // 6)['weight'].nth(0)
    gender = Input.groupby(Input.index // 6)['gender'].nth(0)
    for i in range(0, 6):
        creatinine['creatinine_seq_' + str(i)] = Input.groupby(Input.index // 6)['creatinine'].nth(i)
        intake['intake_seq_' + str(i)] = Input.groupby(Input.index // 6)['intake'].nth(i)
        urine['urine_seq_' + str(i)] = Input.groupby(Input.index // 6)['urine'].nth(i)
        pH['pH_seq_' + str(i)] = Input.groupby(Input.index // 6)['pH'].nth(i)
        Hct['Hct_seq_' + str(i)] = Input.groupby(Input.index // 6)['Hct'].nth(i)
        BUN['Bun_seq_' + str(i)] = Input.groupby(Input.index // 6)['BUN'].nth(i)
        Na['Na_seq_' + str(i)] = Input.groupby(Input.index // 6)['Na'].nth(i)
        K['K_seq_' + str(i)] = Input.groupby(Input.index // 6)['K'].nth(i)
        TP['TP_seq_' + str(i)] = Input.groupby(Input.index // 6)['TP'].nth(i)
        systolicBP['systolicBP_seq_' + str(i)] = Input.groupby(Input.index // 6)['systolicBP'].nth(i)
        meanBP['meanBP_seq_' + str(i)] = Input.groupby(Input.index // 6)['meanBP'].nth(i)
        MSI['MSI_seq_' + str(i)] = Input.groupby(Input.index // 6)['MSI'].nth(i)
        eGFR['eGFR_seq_' + str(i)] = Input.groupby(Input.index // 6)['eGFR'].nth(i)

    xgb_input = pd.DataFrame()
    xgb_input['age'] = age
    # xgb_input['day']=day
    xgb_input['weight'] = weight
    xgb_input['gender'] = gender
    xgb_input = pd.concat(
        [xgb_input, creatinine, intake, urine, pH, Hct, BUN, Na, K, TP, systolicBP, meanBP, MSI, eGFR], axis=1)
    return xgb_input

def PrepareOutput(Input,PredictRes):
    res=[]
    for i in range(0,len(Input),6):
        DictTemp = {
            'patientID':Input['icustay_id'][i],
            'age':round(Input['age'][i],2).tolist(),
            # 'day':round(Input['day'][i]).tolist(),
            'weight':round(Input['weight'][i],2).tolist(),
            'gender':Input['gender'][i].tolist(),
            'crt':round(Input['creatinine'][i:i+6],2).tolist(),
            'intake':round(Input['intake'][i:i+6],2).tolist(),
            'urine':round(Input['urine'][i:i+6],2).tolist(),
            'pH':round(Input['pH'][i:i+6],2).tolist(),
            'Hct':round(Input['Hct'][i:i+6],2).tolist(),
            'BUN':round(Input['BUN'][i:i+6],2).tolist(),
            'Na':round(Input['Na'][i:i+6],2).tolist(),
            'K':round(Input['K'][i:i+6],2).tolist(),
            'TP':round(Input['TP'][i:i+6],2).tolist(),
            'systolicBP':round(Input['systolicBP'][i:i+6],2).tolist(),
            'meanBP':round(Input['meanBP'][i:i+6],2).tolist(),
            'MSI':round(Input['MSI'][i:i+6],2).tolist(),
            'eGFR':round(Input['eGFR'][i:i+6],2).tolist(),
            'aki':PredictRes[int(i/6)],
        }
        res.append(DictTemp)
    return res

def buildTrain(train, pastDay, futureDay, hours):
    # 處理時序性Data
    X_train = []
    print(train.shape[0])
    train_data = np.array(train)
    #train_x_list=train_data.tolist()#list
    data = []
    index = 0
    for i in range(int(len(train_data)/int(hours/4))):
        small_data = []
        for j in range(int(hours/4)):
            small_data.append(train_data[index])
            index = index + 1
        data.append(small_data)
    return np.array(data)

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm



def packageData(train_norm,period):
    dataunit = []
    time_line = []
    newdata = []
    row = 0
    #period = 4 #how many time period in 24hr
    while row < len(train_norm.index.values)-1:
        offset = 0
        time_line = []
        pair = []
        #take pt's feature in every time period -> time_line : [vector t1,vector t2...]
        while offset < period:
            dataunit = []
            #take pt's feature in one time period -> dataunit : vector t
            for column in train_norm.columns:
                dataunit.append(train_norm.iloc[row + offset]['%s'%column])
            time_line.append(dataunit)
            offset += 1
            # pair : [[vector t1,vector t2...], aki]
        pair.append(time_line)
        pair.append('')
        #newdata : [[[vector t1,vector t2...], aki] , [[vector t1,vector t2...], aki] ....]
        newdata.append(pair)
        row += period
    print("how many patients : ",len(newdata))
    return newdata

def startpage(req):
    return render(req, 'startPage.html')

def testunflod(req):
    return render(req, 'testUnflod.html')

@login_required
def homepage(req):
    return render(req, 'homepage.html')

def login(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('/homepage/')

    username = request.POST.get('username', '')
    password = request.POST.get('password', '')

    user = auth.authenticate(username=username, password=password)

    if user is not None and user.is_active:
        auth.login(request, user)
        return HttpResponseRedirect('/homepage/')
    else:
        return render(request, "login.html")


def logout(request):
    auth.logout(request)
    return render(request, "logout.html")

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = User.objects.create_user(username=username, password=password)
        user.save()
        return HttpResponseRedirect('/login/')
    else:
        return render(request, "register.html")

@login_required
def predictsingle(req):
    return render(req, 'predictSingle.html')

@login_required
def predictfile(req):
    return render(req, 'predictFile.html')

@login_required
def predictresultforinput(request):
    Crt = []
    Intake = []
    Urine = []
    pH = []
    Hct = []
    BUN = []
    Na = []
    K = []
    TP = []
    SystolicBP = []
    MeanBP = []
    MSI = []
    eGFR = []

    Age = []
    Weight = []
    Gender = []

    filename = 'static/Xgboost_Seq0606.sav'
    Xgb_model = pickle.load(open(filename, 'rb'))
    if request.method == 'POST':
        if 'OnlyOneData' in request.POST:
            state = 'OnlyOneData'
            Age = request.POST['Age']
            Weight = request.POST['Weight']
            Gender = request.POST['Gender']
            for i in range(0, 6):
                Crtstr = 'Crt_' + str(i)
                Intakestr = 'Intake_' + str(i)
                Urinestr = 'Urine_' + str(i)
                pHstr = 'pH_' + str(i)
                Hctstr = 'Hct_' + str(i)
                BUNstr = 'BUN_' + str(i)
                Nastr = 'Na_' + str(i)
                Kstr = 'K_' + str(i)
                TPstr = 'TP_' + str(i)
                SystolicBPstr = 'SystolicBP_' + str(i)
                MeanBPstr = 'MeanBP_' + str(i)
                MSIstr = 'MSI_' + str(i)
                eGFRstr = 'eGFR_' + str(i)

                Crt.append(request.POST[Crtstr])
                Intake.append(request.POST[Intakestr])
                Urine.append(request.POST[Urinestr])
                pH.append(request.POST[pHstr])
                Hct.append(request.POST[Hctstr])
                BUN.append(request.POST[BUNstr])
                Na.append(request.POST[Nastr])
                K.append(request.POST[Kstr])
                TP.append(request.POST[TPstr])
                SystolicBP.append(request.POST[SystolicBPstr])
                MeanBP.append(request.POST[MeanBPstr])
                MSI.append(request.POST[MSIstr])
                eGFR.append(request.POST[eGFRstr])

            feature = {'age': Age, 'weight': Weight, 'gender': Gender,
                      'creatinine': Crt, 'intake': Intake, 'urine': Urine,
                      'pH': pH, 'Hct': Hct, 'BUN': BUN, 'Na': Na, 'K': K,
                       'TP': TP, 'systolicBP': SystolicBP, 'meanBP': MeanBP,
                       'MSI': MSI, 'eGFR': eGFR}

            InputData = pd.DataFrame(feature)
            InputData['gender'] = InputData['gender'].map({'F': '0', 'M': '1'})
            InputData = InputData.replace(r'^\s*$', np.nan, regex=True)
            InputData = InputData.fillna(-10000)
            InputData = InputData.astype('float64')
            X_test = ConvertFormat(InputData)
            Y = Xgb_model.predict(X_test.to_numpy())
            Y = Y.tolist()
            print(Y)

        context = {
            'aki': Y[0],
            'feature': feature,
        }
    return render(request, "PredictResultForInput.html", context)

@login_required
def lstmpredictresultforinput(request):
    Crt = []
    Intake = []
    Urine = []
    pH = []
    Hct = []
    BUN = []
    Na = []
    K = []
    TP = []
    SystolicBP = []
    MeanBP = []
    MSI = []
    eGFR = []

    Age = []
    Weight = []
    Gender = []

    lstm_model = load_model('./static/rnn_test.h5')
    if request.method == 'POST':
        if 'OnlyOneData' in request.POST:
            state = 'OnlyOneData'
            Age = request.POST['Age']
            Weight = request.POST['Weight']
            Gender = request.POST['Gender']
            for i in range(0, 6):
                Crtstr = 'Crt_' + str(i)
                Intakestr = 'Intake_' + str(i)
                Urinestr = 'Urine_' + str(i)
                pHstr = 'pH_' + str(i)
                Hctstr = 'Hct_' + str(i)
                BUNstr = 'BUN_' + str(i)
                Nastr = 'Na_' + str(i)
                Kstr = 'K_' + str(i)
                TPstr = 'TP_' + str(i)
                SystolicBPstr = 'SystolicBP_' + str(i)
                MeanBPstr = 'MeanBP_' + str(i)
                MSIstr = 'MSI_' + str(i)
                eGFRstr = 'eGFR_' + str(i)

                Crt.append(request.POST[Crtstr])
                Intake.append(request.POST[Intakestr])
                Urine.append(request.POST[Urinestr])
                pH.append(request.POST[pHstr])
                Hct.append(request.POST[Hctstr])
                BUN.append(request.POST[BUNstr])
                Na.append(request.POST[Nastr])
                K.append(request.POST[Kstr])
                TP.append(request.POST[TPstr])
                SystolicBP.append(request.POST[SystolicBPstr])
                MeanBP.append(request.POST[MeanBPstr])
                MSI.append(request.POST[MSIstr])
                eGFR.append(request.POST[eGFRstr])

            feature = {'age': Age, 'weight': Weight, 'gender': Gender,
                      'creatinine': Crt, 'intake': Intake, 'urine': Urine,
                      'pH': pH, 'Hct': Hct, 'BUN': BUN, 'Na': Na, 'K': K,
                       'TP': TP, 'systolicBP': SystolicBP, 'meanBP': MeanBP,
                       'MSI': MSI, 'eGFR': eGFR}

            InputData = pd.DataFrame(feature)
            InputData['gender'] = InputData['gender'].map({'F': '0', 'M': '1'})
            InputData = InputData.replace(r'^\s*$', np.nan, regex=True)
            InputData = InputData.fillna(-10000)
            InputData = InputData.astype('float64')

            test_norm = normalize(InputData.drop(columns=['icustay_id']))
            test_norm = test_norm.fillna(-1)
            X_test = buildTrain(test_norm, 1, 1, 24)


            Y_test = lstm_model.predict(X_test)
            Y = []
            for i in range(0, Y_test.shape[0]):
                if (Y_test[i][5][0] > 0.5):
                    Y.append(1)
                else:
                    Y.append(0)
            Y = Y.tolist()



        context = {
            'aki': Y[0],
            'feature': feature,
        }
    return render(request, "PredictResultForInput.html", context)


@login_required
def predictresultforfile(request):
    if request.method == 'POST':
        if 'Upload' in request.POST:
            state = 'Upload'
            for filename in os.listdir("./static/media"):
                file = "./static/media/" + filename
                os.remove(file)
            UploadFile = request.FILES['document']
            fs = FileSystemStorage()
            fs.save('./static/media/File', UploadFile)

    File_df = pd.read_csv('./static/media/File', sep=',', header=0, encoding='utf-8')
    icustay_id = File_df.groupby(File_df.index // 6)['icustay_id'].nth(0).tolist()
    X_test = ConvertFormat(File_df)
    X_test= X_test.fillna(-10000)

    filename = './static/Xgboost_Seq0606.sav'
    Xgb_model = pickle.load(open(filename, 'rb'))
    Y = Xgb_model.predict(X_test.to_numpy())
    Y = Y.tolist()
    res = PrepareOutput(File_df, Y)
    pageNo = np.arange(1, len(Y) + 1).tolist()
    temp = np.array([Y, icustay_id, pageNo])
    ViewTable = []
    for i in range(0, temp.shape[1]):
        ViewTable.append(temp[:, i].tolist())
    ViewTable = list(chunks(ViewTable, 4))


    paginator = Paginator(res, 1)
    page = request.GET.get('page')
    contacts = paginator.get_page(page)

    context = {
        'contacts': contacts,
        'all': Y,
        'ViewTable': ViewTable,
    }
    return render(request, "PredictResultForFile.html", context)

def lstmpredictresultforfile(request):
    if request.method == 'POST':
        if 'Upload' in request.POST:
            state = 'Upload'
            for filename in os.listdir("./static/media"):
                file = "./static/media/" + filename
                os.remove(file)
            UploadFile = request.FILES['document']
            fs = FileSystemStorage()
            fs.save('./static/media/File', UploadFile)

    File_df = pd.read_csv('./static/media/File', sep=',', header=0, encoding='utf-8')
    icustay_id = File_df.groupby(File_df.index // 6)['icustay_id'].nth(0).tolist()
    test_norm = normalize(File_df.drop(columns=['icustay_id']))
    test_norm = test_norm.fillna(-1)
    X_test = buildTrain(test_norm, 1, 1, 24)


    lstm_model = load_model('./static/rnn_test.h5')
    Y_test = lstm_model.predict(X_test)
    Y = []
    for i in range(0, Y_test.shape[0]):
        if (Y_test[i][5][0] > 0.5):
            Y.append(1)
        else:
            Y.append(0)
    Y = Y.tolist()
    res = PrepareOutput(File_df, Y)
    pageNo = np.arange(1, len(Y) + 1).tolist()
    temp = np.array([Y, icustay_id, pageNo])
    ViewTable = []
    for i in range(0, temp.shape[1]):
        print(i)
        ViewTable.append(temp[:, i].tolist())
    ViewTable = list(chunks(ViewTable, 4))


    paginator = Paginator(res, 1)
    page = request.GET.get('page')
    contacts = paginator.get_page(page)

    context = {
        'contacts': contacts,
        'all': Y,
        'ViewTable': ViewTable,
    }

    return render(request, "PredictResultForFile.html", context)

