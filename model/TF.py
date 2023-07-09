from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score,roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

# Define the training function : 

def train_network(model, optimizer, loss_function, num_epochs, batch_size, X_train, Y_train, lambda_L1 = 0.0) :
    loss_across_epochs = []
    for epoch in range(num_epochs):
        train_loss= 0.0
        #Explicitly start model training
        model.train()
        for i in range(0,X_train.shape[0],batch_size):
            #Extract train batch from X and Y
            input_data = X_train[i:min(X_train.
            shape[0],i+batch_size)]
            labels = Y_train[i:min(X_train.shape[0],i+batch_size)]
            #set the gradients to zero before starting to do backpropragation
            optimizer.zero_grad()
            #Forward pass
            output_data = model(input_data)
            #Caculate loss
            loss = loss_function(output_data, labels)
            L1_loss = 0
            #Compute L1 penalty to be added with loss
            for p in model.parameters():
                L1_loss = L1_loss + p.abs().sum()

            #Add L1 penalty to loss
            loss = loss + lambda_L1 * L1_loss
            #Backpropogate
            loss.backward()
            #Update weights
            optimizer.step()
            train_loss += loss.item() * input_data.size(0)
        loss_across_epochs.append(train_loss/X_train.size(0))
        if epoch%100 == 0:
            print("Epoch: {} - Loss:{:.4f}".format(epoch, train_loss/X_train.size(0)))
    return(loss_across_epochs)

# Defining the Function to Evaluate the Model Performance

def evaluate_model(model,x_test,y_test,X_train,Y_train,loss_list):
    model.eval() #Explicitly set to evaluate mode
    #Predict on Train and Validation Datasets
    y_test_prob = model(x_test)
    y_test_pred =np.where(y_test_prob>0.5,1,0)
    Y_train_prob = model(X_train)
    Y_train_pred =np.where(Y_train_prob>0.5,1,0)
    #Compute Training and Validation Metrics
    print("\n Model Performance -")
    print("Training Accuracy-",round(accuracy_score(Y_train, Y_train_pred),3))
    print("Training Precision-",round(precision_score (Y_train,Y_train_pred),3))
    print("Training Recall-",round(recall_score(Y_train, Y_train_pred),3))
    print("Training ROCAUC", round(roc_auc_score(Y_train,Y_train_prob.detach().numpy()),3))

    print("Validation Accuracy-",round(accuracy_score(y_test, y_test_pred),3))
    print("Validation Precision-",round(precision_score(y_test, y_test_pred),3))
    print("Validation Recall-",round(recall_score(y_test, y_test_pred),3))
    print("Validation ROCAUC", round(roc_auc_score(y_test,y_test_prob.detach().numpy()),3))
    print("\n")

    #Plot the Loss curve and ROC Curve
    plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title('Loss across epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.subplot(1, 2, 2)
    #Validation
    fpr_v, tpr_v, _ = roc_curve(y_test, y_test_prob.detach().numpy())
    roc_auc_v = auc(fpr_v, tpr_v)
    #Training
    fpr_t, tpr_t, _ = roc_curve(Y_train, Y_train_prob.detach().numpy())
    roc_auc_t = auc(fpr_t, tpr_t)
    plt.title('Receiver Operating Characteristic:Validation')
    plt.plot(fpr_v, tpr_v, 'b', label = 'Validation AUC = %0.2f' % roc_auc_v)
    plt.plot(fpr_t, tpr_t, 'r', label = 'Training AUC = %0.2f' % roc_auc_t)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()