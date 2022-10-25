import matplotlib.pyplot as plt

def plot_model_history(history):
    
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history['accuracy'], label='test accuracy')
    axs[0].plot(history.history['val_accuracy'], label='validation accuracy')
    axs[0].legend(loc='lower right')
    
    
    axs[1].plot(history.history['loss'], label='test loss')
    axs[1].plot(history.history['val_loss'], label='validation loss')
    axs[1].legend(loc='upper right')

    plt.show()