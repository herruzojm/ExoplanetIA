import numpy as np
import matplotlib.pyplot as plt
import torch

# Extraemos una proporcion aleatoria de los datos de entrenamiento, primero de los casos positivos, 
# luego de los negativos, y los juntamos para obtener df de entrenamiento y validacion 
def split_train_df(df, proportion):
    df_exoplanet = df[df.LABEL.eq(1)]
    df_no_exoplanet = df[df.LABEL.eq(0)]
    
    exoplanet_indexes = np.random.choice(df_exoplanet.index, int(len(df_exoplanet)*proportion), replace=False)
    df_validation = df_exoplanet.loc[exoplanet_indexes]
    df_train = df_exoplanet.drop(exoplanet_indexes)
    
    no_exoplanet_indexes = np.random.choice(df_no_exoplanet.index, int(len(df_no_exoplanet)*proportion), replace=False)
    df_validation = df_validation.append(df_no_exoplanet.loc[no_exoplanet_indexes])
    df_train = df_train.append(df_no_exoplanet.drop(no_exoplanet_indexes))
    
    return df_train, df_validation

# Separamos la columna con la variable dependiente
def generate_x_y_df(df):
    return df.drop('LABEL', axis = 1), df['LABEL'].copy()

# Normaliza los valores al rango [0, 1], pero es muy sensible a valores extremos
def min_max_scaling(df):
    df_normalized = (df - df.min()) / (df.max() - df.min())
    df_normalized['LABEL'] = df['LABEL']
    return df_normalized

# Menos sensible a valores extremos, pero no garantiza un rango fijo
def z_score_normalizing(df):
    df_normalized = (df - df.mean())/df.std()  
    df_normalized['LABEL'] = df['LABEL']
    return df_normalized


# Mostramos la grafica de luz
def show_flux_plot(df, indexes):
    for i in indexes:
        flux = df.iloc[i,:]
        # 80 days / 3197 columns = 36 minutes
        time = np.arange(len(flux)) * (36.0/60.0) # plotting each hour
        plt.figure(figsize=(15,5))
        plt.title('Flux of star {}'.format(i+1))
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
def calculate_score(y, pred, alpha, beta, imprimir = False):
    tp = torch.sum(pred * y)
    fp = torch.sum(pred * (1 - y))
    fn = torch.sum((1 - pred) * y)
    tn = torch.sum((1 - pred) * (1 - y))
    
    if imprimir:
        print('Matriz de confusión:')
        print('\t\t\tPredicciones')
        print('Valor real\tNegativos\tPositivos')
        print('Negativos\t{}\t\t{}'.format(tn, fp))
        print('Positivos\t{}\t\t{}'.format(fn, tp))
        print()
        
    acierto = (tp + tn).float() / len(y)
    sensibilidad = (tp).float() / (tp + fn)
    especificidad = (tn).float() / (tn + fp)
        
    score = acierto * (alpha * sensibilidad + beta * especificidad)

    if imprimir:
        print('Acierto: {}'.format(acierto),
              'Sensibilidad: {}'.format(sensibilidad),
              'Especificidad: {}'.format(especificidad),
              'Score: {}'.format(score))

    return score

def train(modelo, criterion, optimizer, epochs, alpha, beta, train_x, train_y, validation_x = None, validation_y = None):
    train_losses = [] 
    validation_losses = []
    scores = []
    
    validation = False
    if validation_x != None and validation_y != None:
        validation = True
        
    for epoch in range(epochs):
        optimizer.zero_grad()        
        predictions = modelo(train_x)
        loss = criterion(predictions.squeeze(), train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)
        scores.append(calculate_score(train_y, torch.argmax(predictions, 1), alpha, beta))
            
        if validation:
            modelo.eval()
            predictions = modelo(validation_x)
            validation_loss = criterion(predictions.squeeze(), validation_y)
            modelo.train()
            validation_losses.append(validation_loss)
            
        if epoch % 10 == 0:
            if validation:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(loss.item()),
                     'Validation loss {}'.format(validation_loss.item()))
            else:
                print('Epoch: {}'.format(epoch),
                     'Train loss {}'.format(loss.item()))


    if validation:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(loss.item()),
             'Validation loss {}'.format(validation_loss.item()))
    else:
        print('Epoch: {}'.format(epoch),
             'Train loss {}'.format(loss.item()))
    
    return train_losses, validation_losses, scores