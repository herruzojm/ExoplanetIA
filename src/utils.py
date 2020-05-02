import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from scipy import ndimage, fft
from sklearn.metrics import roc_curve, auc
from fluxdataset import *
from resultados import *
    
# Extraemos una proporcion aleatoria de los datos de entrenamiento, primero de los casos positivos, 
# luego de los negativos, y los juntamos para obtener df de entrenamiento y validacion 
def split_train_df(df, proportion, seed = 19881):
    df_exoplanet = df[df.LABEL.eq(1)]
    df_no_exoplanet = df[df.LABEL.eq(0)]
    
    exoplanet_indexes = np.random.choice(df_exoplanet.index, int(len(df_exoplanet)*proportion), replace=False)
    df_validation = df_exoplanet.loc[exoplanet_indexes]
    df_train = df_exoplanet.drop(exoplanet_indexes)
    
    no_exoplanet_indexes = np.random.choice(df_no_exoplanet.index, int(len(df_no_exoplanet)*proportion), replace=False)
    df_validation = df_validation.append(df_no_exoplanet.loc[no_exoplanet_indexes])
    df_train = df_train.append(df_no_exoplanet.drop(no_exoplanet_indexes))
    
    return df_train.sample(frac = 1), df_validation.sample(frac = 1) 

# Separamos la columna con la variable dependiente
def generate_x_y_df(df):
    return df.drop('LABEL', axis = 1), df['LABEL'].copy()

# Normaliza los valores al rango [0, 1], pero es muy sensible a valores extremos
def min_max_scaling(df):
    return (df - df.min()) / (df.max() - df.min())

# Menos sensible a valores extremos, pero no garantiza un rango fijo
def z_score_normalizing(df):
    return (df - df.mean()) / df.std()  

# Suavizado de la señal con filtro gaussiano
def gaussian_filter(df, substract):
    if substract:
        return df - ndimage.gaussian_filter(df, sigma = 10)
    return ndimage.gaussian_filter(df, sigma = 10)
  
# Transformamos la intensidad en frecuencia mediante Fourier
def fourier_transformation(df):
    return np.abs(fft(df))    
    
# Mostramos la grafica de luz
def show_flux_plot(df, indexes, text = 'Flux'):
    for i in indexes:
        flux = df.iloc[i,:]
        # 80 days / 3197 columns = 36 minutes
        time = np.arange(len(flux)) * (36.0/60.0) # plotting each hour
        plt.figure(figsize=(15,5))
        plt.title('{} of star {}'.format(text, i+1))
        plt.ylabel('Flux, e-/s')
        plt.xlabel('Time, hours')
        plt.plot(time, flux)
        plt.show()

def reduce_upper_outliers(df, reduce = 0.01, half_width = 4):
    '''
    Función adaptada de 
    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration#Data-Processing
    
    reduce = proporción de puntos a reducir
    half_width = número de puntos a cada lado del punto a reducir
    
    Dado que estamos buscando disminuciones en la intensidad de la luz, vamos a eliminar los picos
    de subida de la luz que pueden dificultar el entrenamiento y que son producto de mediciones
    erróneas u otros sucesos físicos distintos.
    
    Esta función busca un porcentaje de esos puntos y los reduce al valor de medio de los puntos
    a su alrededor
    
    '''
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        for j in range(remove):
            idx = sorted_values.index[j]
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)]
                
                count += 1
            new_val /= count # count will always be positive here
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                #df.set_value(i,idx,new_val)
                df.at[i, idx] = new_val
                    
    return df

 # Calcula el score en base a la sensibilidad y la especificidad
def calculate_score(y, pred, alpha, beta, best_previous_score):
    tp = torch.sum(pred * y)
    fp = torch.sum(pred * (1 - y))
    fn = torch.sum((1 - pred) * y)
    tn = torch.sum((1 - pred) * (1 - y))    
        
    accuracy = (tp + tn).float() / len(y)
    recall = (tp).float() / (tp + fn)
    specificity = (tn).float() / (tn + fp)
        
    score = accuracy * (alpha * recall + beta * specificity)

    if score > best_previous_score:
        print('Matriz de confusión:')
        print('\t\t\tPredicciones')
        print('Valor real\tNegativos\tPositivos')
        print('Negativos\t{}\t\t{}'.format(tn, fp))
        print('Positivos\t{}\t\t{}'.format(fn, tp))
        print()
        
        print('Acierto: {}'.format(accuracy),
              'Sensibilidad: {}'.format(recall),
              'Especificidad: {}'.format(specificity),
              'Score: {}'.format(score))
       
    return score

# Calcula las tasas de verdaderos positivos y negativos asi como el area bajo la curva
def calculate_roc(y, pred):
    y_cpu, pred_cpu = y.cpu(), pred.cpu()
    fpr, tpr, _ = roc_curve(y_cpu, pred_cpu)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

#Entrena modelos con salida de dos neuronas
def train_cross(modelo, model_name, criterion, optimizer, epochs, alpha, beta, df_train, df_validation = None, device = "cpu"):
    t = time.perf_counter()
    train_losses = [] 
    validation_losses = []
    scores = []
    best_score = 0
    
    train_dataset = FluxDataset(df_train, device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = True)
    
    validation = False
    if df_validation is not None:
        validation = True
        validation_x, validation_y = generate_x_y_df(df_validation)
        validation_x_tensor = torch.tensor(validation_x.values, device = device).float()
        validation_y_tensor = torch.tensor(validation_y.values, device = device)
    
    for epoch in range(epochs):
        train_loss = 0
        modelo.train()
        
        for target, sample in train_dataloader:
            optimizer.zero_grad()
            predictions = modelo(sample)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_dataloader))        

        if validation:
            modelo.eval()
            predictions = modelo(validation_x_tensor)
            validation_loss = criterion(predictions.squeeze(), validation_y_tensor)
            validation_losses.append(validation_loss.item()) 
            score = calculate_score(validation_y_tensor, torch.argmax(predictions, 1), alpha, beta)
            scores.append(score)
            print('Score {} at epoch {}'.format(score, epoch))
            if score > best_score:
                print('New model saved')
                best_score = score
                torch.save(modelo.state_dict(), '{}.pth'.format(model_name))
                score = calculate_score(validation_y_tensor, torch.argmax(predictions, 1), alpha, beta, True)
                print()

        if epoch % 1 == 0:
            if validation:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(train_losses[-1]),
                     'Validation loss {}'.format(validation_loss.item()))
            else:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(loss.item()))


    if validation:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(train_losses[-1]),
             'Validation loss {}'.format(validation_loss.item()))
    else:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(loss.item()))
    
    print('Best score {}'.format(max(scores)))
    execution_time = time.perf_counter() - t
    print('execution time {}'.format(execution_time))
    
    return train_losses, validation_losses, scores

#Entrena modelos con salida de una neurona
def train_bce(modelo, model_name, criterion, optimizer, epochs, alpha, beta, df_train, df_validation = None, device = "cpu"):
    t = time.perf_counter()    
    train_losses = [] 
    validation_losses = []
    scores = []
    best_score = 0
    
    train_dataset = FluxDataset(df_train, device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = True)
    
    validation = False
    if df_validation is not None:
        validation = True
        validation_x, validation_y = generate_x_y_df(df_validation)
        validation_x_tensor = torch.tensor(validation_x.values, device = device).float()
        validation_y_tensor = torch.tensor(validation_y.values, device = device)
    
    for epoch in range(epochs):
        train_loss = 0
        modelo.train()
        
        for target, sample in train_dataloader:
            optimizer.zero_grad()
            predictions = modelo(sample)
            predictions = predictions[0]
            loss = criterion(predictions, target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/len(train_dataloader))        

        if validation:
            modelo.eval()
            predictions = modelo(validation_x_tensor)
            validation_loss = criterion(predictions.squeeze(), validation_y_tensor.float())
            validation_losses.append(validation_loss.item()) 
            score = calculate_score(validation_y_tensor, torch.argmax(predictions, 1), alpha, beta)
            scores.append(score)
            print('Score {} at epoch {}'.format(score, epoch))
            if score > best_score:
                print('New model saved')
                best_score = score
                torch.save(modelo.state_dict(), '{}.pth'.format(model_name))
                score = calculate_score(validation_y_tensor, torch.argmax(predictions, 1), alpha, beta, True)
                print()

        if epoch % 1 == 0:
            if validation:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(train_losses[-1]),
                     'Validation loss {}'.format(validation_loss.item()))
            else:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(loss.item()))


    if validation:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(train_losses[-1]),
             'Validation loss {}'.format(validation_loss.item()))
    else:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(loss.item()))
    
    print('Best score {}'.format(max(scores)))
    execution_time = time.perf_counter() - t
    print('execution time {}'.format(execution_time))
    
    return train_losses, validation_losses, scores

#Entrena modelos lstm
def train_lstm(modelo, model_name, criterion, optimizer, epochs, alpha, beta, df_train, df_validation = None, device = "cpu"):
    t = time.perf_counter()
    resultados = Resultados()
    
    train_dataset = FluxDataset(df_train, device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle = True)
    
    validation = False
    if df_validation is not None:
        validation = True
        validation_x, validation_y = generate_x_y_df(df_validation)
        validation_x_tensor = torch.tensor(validation_x.values, device = device).float()
        validation_y_tensor = torch.tensor(validation_y.values, device = device)
    
    for epoch in range(epochs):
        train_loss = 0
        modelo.train()
        
        for target, sample in train_dataloader:
            optimizer.zero_grad()
            predictions = modelo(sample.view(len(sample), 1, -1))
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        resultados.add_train_loss(train_loss/len(train_dataloader))

        if validation:
            modelo.eval()
            predictions = modelo(validation_x_tensor.view(len(validation_x_tensor), 1, -1))
            validation_loss = criterion(predictions.squeeze(), validation_y_tensor)
            resultados.add_validation_loss(validation_loss.item())
            score = calculate_score(validation_y_tensor, torch.argmax(predictions, 1), alpha, beta, resultados.best_score)            
            if resultados.is_best_score(score):
                print('New model saved')
                torch.save(modelo.state_dict(), '{}.pth'.format(model_name))
                fpr, tpr, roc_auc = calculate_roc(validation_y_tensor, torch.argmax(predictions, 1))
                resultados.set_roc(fpr, tpr, roc_auc)                
            resultados.add_score(score)            
            print('Score {} at epoch {}'.format(score, epoch))

        resultados.print_resultados(epoch)       
    
    resultados.print_best_score()
    execution_time = time.perf_counter() - t
    print('execution time {}'.format(execution_time))
    
    return resultados

# Comprueba el resultado de un modelo de percetron
def test_model(modelo, model_name, df, alpha = 0.5, beta = 0.5):
    modelo.load_state_dict(torch.load('{}.pth'.format(model_name)))
    modelo.eval()
    
    test_x, test_y = generate_x_y_df(df)
    test_x_tensor = torch.tensor(test_x.values).float()
    test_y_tensor = torch.tensor(test_y)
    
    predictions = modelo(test_x_tensor)
    calculate_score(test_y_tensor, torch.argmax(predictions, 1), alpha, beta, True)

# Comprueba el resultado de un modelo lstm
def test_model_lstm(modelo, model_name, df, alpha = 0.5, beta = 0.5):
    modelo.load_state_dict(torch.load('{}.pth'.format(model_name)))
    modelo.eval()
    
    test_x, test_y = generate_x_y_df(df)
    test_x_tensor = torch.tensor(test_x.values).float()
    test_y_tensor = torch.tensor(test_y)
    
    predictions = modelo(test_x_tensor.view(len(test_x_tensor), 1, -1))
    calculate_score(test_y_tensor, torch.argmax(predictions, 1), alpha, beta, True)
