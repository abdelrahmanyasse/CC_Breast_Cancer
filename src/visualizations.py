import matplotlib.pyplot as plt

def Accuracy_loss_visualization(Training):
    fig = plt.figure(figsize=(12, 4))
    
    plt.plot(Training.history['accuracy'])
    plt.plot(Training.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(["training data", 'validations data'], loc='lower right')
    plt.savefig('output/accuracy_plot.png')
    plt.close(fig)
    return fig

def Accuracy_epochs_visualization(Training):
    fig = plt.figure(figsize=(12, 4))
    
    plt.plot(Training.history['loss'])
    plt.plot(Training.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(["training data", 'validations data'], loc='upper right')
    plt.savefig('output/loss_plot.png')
    plt.close(fig)
    return fig